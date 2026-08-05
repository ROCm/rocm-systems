/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace RcclUnitTesting
{

/**
 * @brief Generic thread-safe process isolated test runner
 *
 * This class provides a framework for running tests in isolated processes
 * using fork()+execv(), with per-test environment control and optional
 * parallel execution with GPU-aware slot scheduling.
 *
 */
class ProcessIsolatedTestRunner
{
public:
    /// Env var set before execv(); its presence in a re-exec'd child selects the test to run.
    static constexpr const char* kReexecMarkerEnvVar = "RCCL_PIT_REEXEC_TEST";

    /// Special value for captureProcessOutput timeout: no deadline, wait indefinitely.
    static constexpr int kNoTimeoutSeconds = 0;

    /**
     * @brief Test execution result structure
     */
    struct TestResult
    {
        std::string                                  testName;     ///< Name of the test
        bool                                         passed;       ///< Whether the test passed
        bool                                         skipped;      ///< Whether the test skipped
        int                                          exitCode;     ///< Process exit code
        pid_t                                        processId;    ///< Process ID that ran the test
        std::chrono::milliseconds                    duration;     ///< Test execution duration
        std::string                                  errorMessage; ///< Error message if test failed
        std::unordered_map<std::string, std::string> environment;  ///< Environment variables used

        /**
         * @brief Default constructor
         */
        TestResult();
    };

    /**
     * @brief Test configuration structure
     */
    struct TestConfig
    {
        /// Special value for numGpus: CPU-only test, no GPU slot acquired.
        static constexpr size_t kCpuOnly = 0;

        std::string           name;      ///< Test name
        std::function<void()> testLogic; ///< Test function to execute
        std::unordered_map<std::string, std::string>
                                 environmentVariables; ///< Environment variables to set
        std::chrono::seconds     timeout;              ///< Test timeout
        bool                     inheritParentEnv;     ///< Whether to inherit parent environment
        std::vector<std::string> clearEnvVars; ///< Environment variables to explicitly clear
        size_t                   numGpus; ///< GPU slots needed; see withNumGpus(). Default: kCpuOnly.

        /**
         * @brief Constructor
         * @param testName Name of the test
         * @param logic Test function to execute
         */
        TestConfig(const std::string& testName, std::function<void()> logic);

        /// Set environment variables for this test.
        TestConfig& withEnvironment(const std::unordered_map<std::string, std::string>& env);

        /// Set the wall-clock timeout for this test.
        TestConfig& withTimeout(std::chrono::seconds timeoutSeconds);

        /**
         * @brief Configure environment inheritance.
         * @param cleanEnv true (default) = start with a clean environment;
         *                 false = inherit all parent environment variables.
         */
        TestConfig& withCleanEnvironment(bool cleanEnv = true);

        /// Clear a specific environment variable before the test runs.
        TestConfig& clearVariable(const std::string& varName);

        /// Set a specific environment variable for this test.
        TestConfig& setVariable(const std::string& name, const std::string& value);

        /**
         * @brief Declare how many GPU slots this test requires during parallel execution.
         *
         * The runner partitions the GPU pool into non-overlapping subsets and injects
         * HIP_VISIBLE_DEVICES so concurrent tests never share a physical device.
         *
         * @param n kCpuOnly (0) = no slot acquired, runs freely in parallel (default).
         *          1           = one dedicated GPU from the pool.
         *          N           = exactly N GPUs from the pool.
         */
        TestConfig& withNumGpus(size_t n);
    };

    /**
     * @brief Execution options for test runner
     */
    struct ExecutionOptions
    {
        /// Special value for maxParallelJobs: use GPU pool size as the degree of
        /// parallelism, falling back to std::thread::hardware_concurrency() when
        /// no GPU pool is available.
        static constexpr size_t kAutoParallelism = 0;

        bool   stopOnFirstFailure; ///< Stop execution on first test failure
        bool   verboseLogging;     ///< Enable verbose logging
        size_t maxParallelJobs;    ///< Maximum number of concurrent child processes.
                                   ///< 1 = sequential (default).
                                   ///< kAutoParallelism (0) = GPU pool size, or hardware_concurrency() if no pool.
                                   ///< N > 1 = up to N tests run simultaneously.
                                   ///< Results are always reported in registration order.

        /// Physical GPU device indices available for distribution across parallel
        /// test processes.  Each concurrent child is assigned a non-overlapping
        /// subset and sees only its assigned GPUs via HIP_VISIBLE_DEVICES so
        /// tests never contend on the same device.
        ///
        /// Empty (default): auto-detect via HIP_VISIBLE_DEVICES, then KFD sysfs.
        /// Only used when maxParallelJobs > 1.
        std::vector<int> gpuPool;

        /**
         * @brief Default constructor with sensible defaults
         */
        ExecutionOptions();
    };

private:
    /**
     * @brief Structure to hold captured process output
     */
    struct CapturedOutput
    {
        std::string stdoutContent; ///< Captured stdout content
        std::string stderrContent; ///< Captured stderr content
    };
    // Thread-safe static members for test management
    static std::mutex              testConfigsMutex_;
    static std::vector<TestConfig> testConfigs_;
    static std::mutex              resultsMutex_;
    static std::vector<TestResult> testResults_;

    /**
     * @brief Apply environment variables to current process
     * @param config Test configuration containing environment settings
     */
    static void applyEnvironmentVariables(const TestConfig& config);

    /**
     * @brief Execute a single test in the child process
     * @param config Test configuration
     * @return Exit code (0 for success, non-zero for failure)
     */
    static int runTestInProcess(const TestConfig& config);

    /**
     * @brief Create pipes for capturing process output
     * @param stdoutPipe Array to hold stdout pipe file descriptors [read, write]
     * @param stderrPipe Array to hold stderr pipe file descriptors [read, write]
     * @return True if pipes were created successfully, false otherwise
     */
    static bool createOutputPipes(int stdoutPipe[2], int stderrPipe[2]);

    /**
     * @brief Redirect child process output to pipes
     * @param stdoutPipe Stdout pipe file descriptors [read, write]
     * @param stderrPipe Stderr pipe file descriptors [read, write]
     */
    static void redirectOutputToPipes(int stdoutPipe[2], int stderrPipe[2]);

    /**
     * @brief Capture output from child process via pipes
     * @param stdoutPipe Stdout pipe file descriptors [read, write]
     * @param stderrPipe Stderr pipe file descriptors [read, write]
     * @param pid Child process ID to monitor
     * @param status Pointer to status variable for waitpid
     * @param timeout Wall-clock limit; 0 = unlimited.  On expiry the child is
     *                SIGTERM'd (then SIGKILL'd) and *status is set to indicate
     *                RCCL_TEST_TIMEOUT.
     * @return Captured output from stdout and stderr
     */
    static CapturedOutput captureProcessOutput(
        int                  stdoutPipe[2],
        int                  stderrPipe[2],
        pid_t                pid,
        int*                 status,
        std::chrono::seconds timeout = std::chrono::seconds(kNoTimeoutSeconds)
    );

    /**
     * @brief Display captured output with formatted delimiters
     * @param output Captured output to display
     * @param testName Name of the test for context
     */
    static void displayCapturedOutput(const CapturedOutput& output, const std::string& testName);

    /**
     * @brief Handle re-exec child entrypoint
     *
     * If kReexecMarkerEnvVar is set, this process is a re-exec child:
     * run the matching test lambda, flush coverage, and _exit().
     * Returns normally only when called from the original parent process.
     *
     * @param tests Test configurations to search for the target
     */
    static void handleReexecEntrypoint(const std::vector<TestConfig>& tests);

    /// Return type for a single spawned test: result + captured output.
    struct SpawnOutcome
    {
        TestResult     result;
        CapturedOutput output;
    };

    /// Callable type for spawning one isolated test.
    using SpawnFn = std::function<SpawnOutcome(const TestConfig&, const std::vector<int>&)>;

    /**
     * @brief Run tests one at a time (no GPU slot management).
     */
    static void runSequential(
        const std::vector<TestConfig>& tests,
        const ExecutionOptions&        opts,
        const SpawnFn&                 spawnFn);

    /**
     * @brief Run tests with a bounded sliding window and GPU slot management.
     * @param parallelism Maximum simultaneous child processes.
     * @param gpuPool     Physical GPU indices available for slot assignment.
     */
    static void runParallel(
        const std::vector<TestConfig>& tests,
        const ExecutionOptions&        opts,
        size_t                         parallelism,
        const std::vector<int>&        gpuPool,
        const SpawnFn&                 spawnFn);

    /**
     * @brief Generate and display test report
     * @param options          Execution options used for the test run
     * @param totalRegistered  Total number of tests registered (may differ from
     *                         the number run when stopOnFirstFailure is active)
     * @return True if no tests failed
     */
    static bool generateReport(const ExecutionOptions& options, size_t totalRegistered);

public:
    /**
     * @brief Register a test configuration
     * @param config Complete test configuration
     */
    static void registerTest(const TestConfig& config);

    /**
     * @brief Register a simple test with just name and logic
     * @param name Test name
     * @param testLogic Test function to execute
     */
    static void registerTest(const std::string& name, std::function<void()> testLogic);

    /**
     * @brief Register a test with environment variables
     * @param name Test name
     * @param testLogic Test function to execute
     * @param env Environment variables to set for this test
     */
    static void registerTest(
        const std::string&                                  name,
        std::function<void()>                               testLogic,
        const std::unordered_map<std::string, std::string>& env
    );

    /**
     * @brief Record a test result (thread-safe)
     * @param result Test result to record
     */
    static void recordTestResult(const TestResult& result);

    /**
     * @brief Execute all registered tests (sequentially or in parallel)
     * @param options Execution options (defaults to sequential, continue on failure)
     * @return True if all tests passed, false if any failed or the GPU pool could not be detected
     * @note This method automatically clears all test registrations and results
     *       after execution, ensuring a clean state for the next test suite.
     */
    static bool executeAllTests(const ExecutionOptions& options = ExecutionOptions());

    /**
     * @brief Get detailed test results (thread-safe)
     * @return Vector of all test results
     */
    static std::vector<TestResult> getTestResults();

    /**
     * @brief Clear test registry and results (thread-safe)
     * @note Calling this method manually is typically not necessary, as
     *       executeAllTests() automatically clears registrations after execution.
     *       This method is primarily useful for advanced use cases or when tests
     *       are registered but not executed.
     */
    static void clear();

    /**
     * @brief Get number of registered tests
     * @return Number of registered tests
     */
    static size_t getTestCount();
};

// Macros for Simplified Usage

/**
 * @brief Register and execute a single isolated test with minimal boilerplate
 *
 * Uses variadic macros to automatically handle commas in lambda bodies
 *
 * @param test_name Name of the test (string)
 * @param ... Lambda containing test logic (variadic to handle internal commas)
 *
 * Example:
 *   RUN_ISOLATED_TEST("MyTest", []() {
 *     EXPECT_TRUE(someFunction());
 *   });
 */
#define RUN_ISOLATED_TEST(test_name, ...)                                                   \
    do                                                                                      \
    {                                                                                       \
        ::RcclUnitTesting::ProcessIsolatedTestRunner::registerTest(test_name, __VA_ARGS__); \
        bool passed_ = ::RcclUnitTesting::ProcessIsolatedTestRunner::executeAllTests();     \
        EXPECT_TRUE(passed_) << "Isolated test '" << test_name << "' failed";               \
    }                                                                                       \
    while(0)

/**
 * @brief Register and execute a single isolated test with environment variables
 *
 * Uses variadic macros to automatically handle environment variable initializer lists
 *
 * @param test_name Name of the test (string)
 * @param test_body Lambda containing test logic
 * @param ... Environment variables as initializer list
 *
 * Example:
 *   RUN_ISOLATED_TEST_WITH_ENV("MyTest",
 *     []() { EXPECT_TRUE(someFunction()); },
 *     {{"VAR1", "value1"}, {"VAR2", "value2"}});
 *
 * Note: Uses __VA_ARGS__ to capture environment variables, which automatically
 * handles commas in the initializer list without requiring extra parentheses.
 */
#define RUN_ISOLATED_TEST_WITH_ENV(test_name, test_body, ...)                           \
    do                                                                                  \
    {                                                                                   \
        ::RcclUnitTesting::ProcessIsolatedTestRunner::registerTest(                     \
            test_name,                                                                  \
            test_body,                                                                  \
            __VA_ARGS__                                                                 \
        );                                                                              \
        bool passed_ = ::RcclUnitTesting::ProcessIsolatedTestRunner::executeAllTests(); \
        EXPECT_TRUE(passed_) << "Isolated test '" << test_name << "' failed";           \
    }                                                                                   \
    while(0)

/**
 * @brief Register and execute multiple isolated tests with default options
 *
 * This macro takes multiple TestConfig objects and executes them all.
 * Tests are automatically cleaned up after execution.
 *
 * Example:
 *   RUN_ISOLATED_TESTS(
 *     ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
 *     ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
 *       .withEnvironment({{"VAR", "value"}}),
 *     ProcessIsolatedTestRunner::TestConfig("Test3", []() { ... })
 *       .withTimeout(std::chrono::seconds(60))
 *   );
 */
#define RUN_ISOLATED_TESTS(...)                                                              \
    do                                                                                       \
    {                                                                                        \
        ::RcclUnitTesting::ProcessIsolatedTestRunner::TestConfig configs_[] = {__VA_ARGS__}; \
        for(const auto& config_ : configs_)                                                  \
        {                                                                                    \
            ::RcclUnitTesting::ProcessIsolatedTestRunner::registerTest(config_);             \
        }                                                                                    \
        bool passed_ = ::RcclUnitTesting::ProcessIsolatedTestRunner::executeAllTests();      \
        EXPECT_TRUE(passed_) << "One or more isolated tests failed";                         \
    }                                                                                        \
    while(0)

/**
 * @brief Register and execute multiple isolated tests with custom options
 *
 * This macro takes execution options and multiple TestConfig objects.
 *
 * Example:
 *   ProcessIsolatedTestRunner::ExecutionOptions opts;
 *   opts.stopOnFirstFailure = true;
 *   opts.verboseLogging = true;
 *
 *   RUN_ISOLATED_TESTS_WITH_OPTIONS(opts,
 *     ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
 *     ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
 *   );
 */
#define RUN_ISOLATED_TESTS_WITH_OPTIONS(options, ...)                                          \
    do                                                                                         \
    {                                                                                          \
        ::RcclUnitTesting::ProcessIsolatedTestRunner::TestConfig configs_[] = {__VA_ARGS__};   \
        for(const auto& config_ : configs_)                                                    \
        {                                                                                      \
            ::RcclUnitTesting::ProcessIsolatedTestRunner::registerTest(config_);               \
        }                                                                                      \
        bool passed_ = ::RcclUnitTesting::ProcessIsolatedTestRunner::executeAllTests(options); \
        EXPECT_TRUE(passed_) << "One or more isolated tests failed";                           \
    }                                                                                          \
    while(0)

} // namespace RcclUnitTesting
