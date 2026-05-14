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
 * with clean environment settings and sequential execution.
 *
 */
class ProcessIsolatedTestRunner
{
public:
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
        std::string           name;      ///< Test name
        std::function<void()> testLogic; ///< Test function to execute
        std::unordered_map<std::string, std::string>
                                 environmentVariables; ///< Environment variables to set
        std::chrono::seconds     timeout;              ///< Test timeout
        bool                     inheritParentEnv;     ///< Whether to inherit parent environment
        std::vector<std::string> clearEnvVars; ///< Environment variables to explicitly clear

        /**
         * @brief Constructor
         * @param testName Name of the test
         * @param logic Test function to execute
         */
        TestConfig(const std::string& testName, std::function<void()> logic);

        /**
         * @brief Set environment variables for this test
         * @param env Map of environment variable name-value pairs
         * @return Reference to this TestConfig for method chaining
         */
        TestConfig& withEnvironment(const std::unordered_map<std::string, std::string>& env);

        /**
         * @brief Set timeout for this test
         * @param timeoutSeconds Timeout in seconds
         * @return Reference to this TestConfig for method chaining
         */
        TestConfig& withTimeout(std::chrono::seconds timeoutSeconds);

        /**
         * @brief Configure environment inheritance
         * @param inherit Whether to inherit parent environment variables
         * @return Reference to this TestConfig for method chaining
         */
        TestConfig& withCleanEnvironment(bool inherit = false);

        /**
         * @brief Clear a specific environment variable
         * @param varName Name of the variable to clear
         * @return Reference to this TestConfig for method chaining
         */
        TestConfig& clearVariable(const std::string& varName);

        /**
         * @brief Set a specific environment variable
         * @param name Variable name
         * @param value Variable value
         * @return Reference to this TestConfig for method chaining
         */
        TestConfig& setVariable(const std::string& name, const std::string& value);
    };

    /**
     * @brief Execution options for test runner
     */
    struct ExecutionOptions
    {
        bool stopOnFirstFailure; ///< Stop execution on first test failure
        bool verboseLogging;     ///< Enable verbose logging

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
     * @return Captured output from stdout and stderr
     */
    static CapturedOutput
        captureProcessOutput(int stdoutPipe[2], int stderrPipe[2], pid_t pid, int* status);

    /**
     * @brief Display captured output with formatted delimiters
     * @param output Captured output to display
     * @param testName Name of the test for context
     */
    static void displayCapturedOutput(const CapturedOutput& output, const std::string& testName);

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
     * @brief Execute all registered tests sequentially
     * @param options Execution options (defaults to continue on failure)
     * @return True if all tests passed, false otherwise
     * @note This method automatically clears all test registrations and results
     *       after execution, ensuring a clean state for the next test suite.
     */
    static bool executeAllTests(const ExecutionOptions& options = ExecutionOptions());

    /**
     * @brief Generate and display test report
     * @param options Execution options used for the test run
     * @return True if all tests passed, false otherwise
     */
    static bool generateReport(const ExecutionOptions& options);

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
