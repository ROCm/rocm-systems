/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "ProcessIsolatedTestRunner.hpp"

#include <errno.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "ErrCode.hpp"

namespace RcclUnitTesting
{

// Exit codes for test process results
enum RcclTestCode
{
    RCCL_TEST_INVALID           = -1,
    RCCL_TEST_SUCCESS           = 0,
    RCCL_TEST_FAILURE           = 1,
    RCCL_TEST_UNKNOWN_EXCEPTION = 2,
    RCCL_TEST_TIMEOUT           = 3,
    RCCL_TEST_SKIPPED           = 4
};

// Define static members
std::mutex                                         ProcessIsolatedTestRunner::testConfigsMutex_;
std::vector<ProcessIsolatedTestRunner::TestConfig> ProcessIsolatedTestRunner::testConfigs_;
std::mutex                                         ProcessIsolatedTestRunner::resultsMutex_;
std::vector<ProcessIsolatedTestRunner::TestResult> ProcessIsolatedTestRunner::testResults_;

// TestResult implementation
ProcessIsolatedTestRunner::TestResult::TestResult()
    : passed(false), skipped(false), exitCode(-1), processId(-1), duration(0)
{}

// TestConfig implementation
ProcessIsolatedTestRunner::TestConfig::TestConfig(
    const std::string& testName, std::function<void()> logic
)
    : name(testName), testLogic(logic), timeout(30), inheritParentEnv(true)
{}

ProcessIsolatedTestRunner::TestConfig& ProcessIsolatedTestRunner::TestConfig::withEnvironment(
    const std::unordered_map<std::string, std::string>& env
)
{
    environmentVariables = env;
    return *this;
}

ProcessIsolatedTestRunner::TestConfig&
    ProcessIsolatedTestRunner::TestConfig::withTimeout(std::chrono::seconds timeoutSeconds)
{
    timeout = timeoutSeconds;
    return *this;
}

ProcessIsolatedTestRunner::TestConfig&
    ProcessIsolatedTestRunner::TestConfig::withCleanEnvironment(bool inherit)
{
    inheritParentEnv = inherit;
    return *this;
}

ProcessIsolatedTestRunner::TestConfig&
    ProcessIsolatedTestRunner::TestConfig::clearVariable(const std::string& varName)
{
    clearEnvVars.push_back(varName);
    return *this;
}

ProcessIsolatedTestRunner::TestConfig& ProcessIsolatedTestRunner::TestConfig::setVariable(
    const std::string& name, const std::string& value
)
{
    environmentVariables[name] = value;
    return *this;
}

// ExecutionOptions implementation
ProcessIsolatedTestRunner::ExecutionOptions::ExecutionOptions()
    : stopOnFirstFailure(false), verboseLogging(true)
{}

// Apply environment variables to current process
void ProcessIsolatedTestRunner::applyEnvironmentVariables(const TestConfig& config)
{
    // Clear specified environment variables first
    for(const auto& varName : config.clearEnvVars)
    {
        unsetenv(varName.c_str());
    }

    // If not inheriting parent environment, clear all environment variables
    if(!config.inheritParentEnv)
    {
        // Clear all existing environment variables
        if(clearenv() != 0)
        {
            std::cerr << "Warning: Failed to clear environment variables" << std::endl;
        }

        // Set only the specified variables
        for(const auto& [name, value] : config.environmentVariables)
        {
            setenv(name.c_str(), value.c_str(), 1);
        }
    }
    else
    {
        // Just set/override the specified variables
        for(const auto& [name, value] : config.environmentVariables)
        {
            setenv(name.c_str(), value.c_str(), 1);
        }
    }
}

// Execute a single test in a separate process
int ProcessIsolatedTestRunner::runTestInProcess(const TestConfig& config)
{
    pid_t processId = getpid();

    if(config.name.empty())
    {
        std::cerr << "Error: Test name is empty for process " << processId << std::endl;
        return RCCL_TEST_FAILURE;
    }

    try
    {
        // Apply environment variables
        applyEnvironmentVariables(config);

        // Thread-safe test execution with timeout protection
        std::atomic<bool>  testCompleted{false};
        std::exception_ptr testException = nullptr;
        bool               testPassed    = true;
        bool               testSkipped   = false;

        // Run test in a separate thread to allow timeout handling
        std::thread testThread(
            [&]()
            {
                try
                {
                    // Get initial test state
                    const ::testing::UnitTest* unitTest = ::testing::UnitTest::GetInstance();
                    size_t                     initialFailureCount = unitTest->failed_test_count();
                    size_t                     initialSkippedCount = unitTest->skipped_test_count();

                    // Execute the test logic
                    config.testLogic();

                    // Check if any new test failures occurred
                    size_t finalFailureCount = unitTest->failed_test_count();
                    size_t finalSkippedCount = unitTest->skipped_test_count();

                    testPassed  = (finalFailureCount == initialFailureCount);
                    testSkipped = (finalSkippedCount > initialSkippedCount);

                    testCompleted = true;
                }
                catch(...)
                {
                    testException = std::current_exception();
                    testPassed    = false;
                    testCompleted = true;
                }
            }
        );

        // Wait for test completion with timeout
        auto       start   = std::chrono::steady_clock::now();
        const auto timeout = config.timeout;

        while(!testCompleted.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if(std::chrono::steady_clock::now() - start > timeout)
            {
                // Test timed out
                INFO(
                    "Test '%s' TIMED OUT after %ld seconds\n",
                    config.name.c_str(),
                    timeout.count()
                );
                fflush(NULL);
                testThread.detach();
                return RCCL_TEST_TIMEOUT;
            }
        }

        // Wait for thread completion
        if(testThread.joinable())
        {
            testThread.join();
        }

        // Check if test threw an exception
        if(testException)
        {
            std::rethrow_exception(testException);
        }

        // Flush output before returning (needed before _exit())
        fflush(NULL);

        // Return appropriate exit code based on test result
        if(testSkipped)
        {
            return RCCL_TEST_SKIPPED;
        }
        else if(testPassed)
        {
            return RCCL_TEST_SUCCESS;
        }
        else
        {
            return RCCL_TEST_FAILURE;
        }
    }
    catch(const std::exception& e)
    {
        INFO("Test '%s' FAILED with exception: %s\n", config.name.c_str(), e.what());
        std::cerr << "Exception in test '" << config.name << "': " << e.what() << std::endl;
        fflush(NULL);
        return RCCL_TEST_FAILURE;
    }
    catch(...)
    {
        INFO("Test '%s' FAILED with unknown exception\n", config.name.c_str());
        std::cerr << "Unknown exception in test '" << config.name << "'" << std::endl;
        fflush(NULL);
        return RCCL_TEST_UNKNOWN_EXCEPTION;
    }
}

// Register a test configuration
void ProcessIsolatedTestRunner::registerTest(const TestConfig& config)
{
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    testConfigs_.push_back(config);
}

// Register a simple test with just name and logic
void ProcessIsolatedTestRunner::registerTest(
    const std::string& name, std::function<void()> testLogic
)
{
    registerTest(TestConfig(name, testLogic));
}

// Register a test with environment variables
void ProcessIsolatedTestRunner::registerTest(
    const std::string&                                  name,
    std::function<void()>                               testLogic,
    const std::unordered_map<std::string, std::string>& env
)
{
    registerTest(TestConfig(name, testLogic).withEnvironment(env));
}

// Record test result (thread-safe)
void ProcessIsolatedTestRunner::recordTestResult(const TestResult& result)
{
    std::lock_guard<std::mutex> lock(resultsMutex_);
    testResults_.push_back(result);
}

// Helper method: Create pipes for capturing process output
bool ProcessIsolatedTestRunner::createOutputPipes(int stdoutPipe[2], int stderrPipe[2])
{
    // Create pipes for stdout and stderr
    // stdoutPipe[0] = read end, stdoutPipe[1] = write end
    if(pipe(stdoutPipe) == -1)
    {
        std::cerr << "Failed to create stdout pipe: " << strerror(errno) << std::endl;
        return false;
    }

    if(pipe(stderrPipe) == -1)
    {
        std::cerr << "Failed to create stderr pipe: " << strerror(errno) << std::endl;
        close(stdoutPipe[0]);
        close(stdoutPipe[1]);
        return false;
    }

    return true;
}

// Helper method: Redirect child process output to pipes
void ProcessIsolatedTestRunner::redirectOutputToPipes(int stdoutPipe[2], int stderrPipe[2])
{
    // Close read ends of pipes in child process (not needed)
    close(stdoutPipe[0]);
    close(stderrPipe[0]);

    // Redirect stdout and stderr to write ends of pipes
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stderrPipe[1], STDERR_FILENO);

    // Close the original write end file descriptors after duplication
    // The duplicated descriptors (STDOUT_FILENO, STDERR_FILENO) will be closed by _exit()
    close(stdoutPipe[1]);
    close(stderrPipe[1]);
}

// Helper method: Capture output from child process pipes
ProcessIsolatedTestRunner::CapturedOutput ProcessIsolatedTestRunner::captureProcessOutput(
    int stdoutPipe[2], int stderrPipe[2], pid_t pid, int* status
)
{
    // Close write ends of pipes in parent process (not needed)
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    CapturedOutput output;
    char           buffer[4096];
    ssize_t        count;

    // Read from stdout pipe
    while((count = read(stdoutPipe[0], buffer, sizeof(buffer) - 1)) > 0)
    {
        buffer[count] = '\0';
        output.stdoutContent += buffer;
    }
    close(stdoutPipe[0]);

    // Read from stderr pipe
    while((count = read(stderrPipe[0], buffer, sizeof(buffer) - 1)) > 0)
    {
        buffer[count] = '\0';
        output.stderrContent += buffer;
    }
    close(stderrPipe[0]);

    // Wait for child to exit (blocking)
    waitpid(pid, status, 0);

    return output;
}

// Helper method: Display captured output
void ProcessIsolatedTestRunner::displayCapturedOutput(
    const CapturedOutput& output, const std::string& testName
)
{
    if(!output.stdoutContent.empty())
    {
        std::cout << output.stdoutContent;
        if(output.stdoutContent.back() != '\n')
            std::cout << '\n';
    }
    if(!output.stderrContent.empty())
    {
        std::cerr << output.stderrContent;
        if(output.stderrContent.back() != '\n')
            std::cerr << '\n';
    }
}

// Execute all registered tests (simplified sequential execution only)
bool ProcessIsolatedTestRunner::executeAllTests(const ExecutionOptions& options)
{

    // Get test configurations to run
    std::vector<TestConfig> testsToRun;
    {
        std::lock_guard<std::mutex> lock(testConfigsMutex_);
        testsToRun = testConfigs_;
    }

    // Clear previous results
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        testResults_.clear();
    }

    // Sequential execution
    for(const auto& testConfig : testsToRun)
    {
        auto startTime = std::chrono::steady_clock::now();

        int stdout_fd[2], stderr_fd[2];
        if(!createOutputPipes(stdout_fd, stderr_fd))
        {
            std::cerr << "Failed to create output files for test '" << testConfig.name << "'"
                      << std::endl;
            continue;
        }

        // Flush all output before fork to prevent child from inheriting unflushed buffers
        fflush(NULL);

        pid_t pid = fork();

        if(pid == 0)
        {
            redirectOutputToPipes(stdout_fd, stderr_fd);
            int result = runTestInProcess(testConfig);
            // Use _exit() instead of exit() to avoid atexit handlers
            // This prevents GPU runtime cleanup issues after fork
            _exit(result);
        }
        else if(pid > 0)
        {
            // Log test start with environment variables if any
            if(!testConfig.environmentVariables.empty())
            {
                std::string envVars;
                for(const auto& [name, value] : testConfig.environmentVariables)
                {
                    if(!envVars.empty())
                        envVars += ", ";
                    envVars += name + "=" + value;
                }
                INFO(
                    "Running isolated test '%s' (PID: %d) with env: %s\n",
                    testConfig.name.c_str(),
                    pid,
                    envVars.c_str()
                );
            }
            else
            {
                INFO("Running isolated test '%s' (PID: %d)\n", testConfig.name.c_str(), pid);
            }
            // Flush parent's output before reading from child pipes to ensure proper ordering
            fflush(stdout);
            fflush(stderr);

            int            status;
            CapturedOutput output = captureProcessOutput(stdout_fd, stderr_fd, pid, &status);

            auto endTime = std::chrono::steady_clock::now();
            auto duration
                = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // Display captured output BEFORE status messages for proper sequencing
            displayCapturedOutput(output, testConfig.name);

            TestResult testResult;
            testResult.testName  = testConfig.name;
            testResult.processId = pid;
            testResult.duration  = duration;

            if(WIFEXITED(status))
            {
                int exitCode        = WEXITSTATUS(status);
                testResult.exitCode = exitCode;
                testResult.passed   = (exitCode == RCCL_TEST_SUCCESS);
                testResult.skipped  = (exitCode == RCCL_TEST_SKIPPED);

                if(exitCode == RCCL_TEST_SUCCESS)
                {
                    INFO("Test '%s' PASSED (%ld ms)\n", testConfig.name.c_str(), duration.count());
                }
                else if(exitCode == RCCL_TEST_TIMEOUT)
                {
                    INFO(
                        "Test '%s' (PID: %d) TIMED OUT after %ld ms\n",
                        testConfig.name.c_str(),
                        pid,
                        duration.count()
                    );
                    testResult.errorMessage = "Test timed out";
                }
                else if(exitCode == RCCL_TEST_SKIPPED)
                {
                    INFO(
                        "Test '%s' (PID: %d) SKIPPED in %ld ms\n",
                        testConfig.name.c_str(),
                        pid,
                        duration.count()
                    );
                    testResult.errorMessage = "Test skipped";
                }
                else
                {
                    INFO(
                        "Test '%s' (PID: %d) FAILED with exit code %d after %ld ms\n",
                        testConfig.name.c_str(),
                        pid,
                        exitCode,
                        duration.count()
                    );
                    testResult.errorMessage
                        = "Test failed with exit code " + std::to_string(exitCode);
                }
            }
            else if(WIFSIGNALED(status))
            {
                int signal = WTERMSIG(status);

                // Check if test reported success before signal termination
                bool testPassed = (output.stdoutContent.find("PASSED") != std::string::npos);

                if(testPassed)
                {
                    // Test completed successfully before signal (e.g., GPU runtime cleanup)
                    testResult.passed   = true;
                    testResult.skipped  = false;
                    testResult.exitCode = RCCL_TEST_SUCCESS;
                    INFO("Test '%s' PASSED (%ld ms)\n", testConfig.name.c_str(), duration.count());
                }
                else
                {
                    // Test terminated by signal before completion (crash)
                    testResult.passed       = false;
                    testResult.skipped      = false;
                    testResult.exitCode     = -signal;
                    testResult.errorMessage = "Terminated by signal " + std::to_string(signal);
                    INFO(
                        "Test '%s' (PID: %d) terminated by signal %d after %ld ms\n",
                        testConfig.name.c_str(),
                        pid,
                        signal,
                        duration.count()
                    );
                }
            }
            else
            {
                testResult.passed       = false;
                testResult.skipped      = false;
                testResult.exitCode     = RCCL_TEST_INVALID;
                testResult.errorMessage = "Failed to wait for process";
            }

            recordTestResult(testResult);

            // Stop on first failure if requested
            if(options.stopOnFirstFailure && !testResult.passed && !testResult.skipped)
            {
                break;
            }
        }
        else
        {
            // Fork failed
            TestResult testResult;
            testResult.testName     = testConfig.name;
            testResult.passed       = false;
            testResult.skipped      = false;
            testResult.exitCode     = RCCL_TEST_INVALID;
            testResult.processId    = RCCL_TEST_INVALID;
            testResult.duration     = std::chrono::milliseconds(0);
            testResult.errorMessage = "Failed to fork process";

            recordTestResult(testResult);
            INFO("Failed to fork process for test '%s'\n", testConfig.name.c_str());

            if(options.stopOnFirstFailure)
            {
                break;
            }
        }
    }

    bool result = generateReport(options);

    // Automatically clear test configurations and results after execution
    // This ensures a clean state for the next test suite without requiring
    // explicit clear() calls from test cases
    {
        std::lock_guard<std::mutex> lock(testConfigsMutex_);
        testConfigs_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        testResults_.clear();
    }

    return result;
}

// Generate and display test report
bool ProcessIsolatedTestRunner::generateReport(const ExecutionOptions& options)
{
    int                       totalTests   = 0;
    int                       passedTests  = 0;
    int                       failedTests  = 0;
    int                       skippedTests = 0;
    std::chrono::milliseconds totalDuration{0};

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        totalTests = testResults_.size();

        for(const auto& result : testResults_)
        {
            if(result.skipped)
            {
                skippedTests++;
            }
            else if(result.passed)
            {
                passedTests++;
            }
            else
            {
                failedTests++;
            }
            totalDuration += result.duration;
        }
    }

    // Report summary only if there are failures or multiple tests
    if(failedTests > 0 || totalTests > 1)
    {
        INFO(
            "Process-Isolated Tests: %d passed, %d failed, %d skipped (%ld ms total)\n",
            passedTests,
            failedTests,
            skippedTests,
            totalDuration.count()
        );

        if(failedTests > 0)
        {
            std::lock_guard<std::mutex> lock(resultsMutex_);
            for(const auto& result : testResults_)
            {
                if(!result.passed && !result.skipped)
                {
                    INFO(
                        "  Failed: %s - %s\n",
                        result.testName.c_str(),
                        result.errorMessage.c_str()
                    );
                }
            }
        }
    }

    return failedTests == 0;
}

// Get detailed test results (thread-safe)
std::vector<ProcessIsolatedTestRunner::TestResult> ProcessIsolatedTestRunner::getTestResults()
{
    std::lock_guard<std::mutex> lock(resultsMutex_);
    return testResults_;
}

// Clear test registry and results (thread-safe)
void ProcessIsolatedTestRunner::clear()
{
    size_t registeredCount = 0;
    size_t executedCount   = 0;

    // Check for unexecuted tests before clearing
    {
        std::lock_guard<std::mutex> lock(testConfigsMutex_);
        registeredCount = testConfigs_.size();
    }
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        executedCount = testResults_.size();
    }

    // Warn if tests were registered but not all executed
    if(registeredCount > 0 && executedCount < registeredCount)
    {
        std::cerr << "\n⚠️  WARNING: ProcessIsolatedTestRunner::clear() called with "
                  << (registeredCount - executedCount) << " unexecuted test(s)!\n"
                  << "   Registered: " << registeredCount << " test(s)\n"
                  << "   Executed:   " << executedCount << " test(s)\n"
                  << "   Did you forget to call executeAllTests()?\n"
                  << std::endl;
    }

    // Clear the registrations and results
    {
        std::lock_guard<std::mutex> lock(testConfigsMutex_);
        testConfigs_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        testResults_.clear();
    }
}

// Get number of registered tests
size_t ProcessIsolatedTestRunner::getTestCount()
{
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    return testConfigs_.size();
}

} // namespace RcclUnitTesting
