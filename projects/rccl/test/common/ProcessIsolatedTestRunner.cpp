/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "ProcessIsolatedTestRunner.hpp"

#include <errno.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <unistd.h>

#include <dirent.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "ErrCode.hpp"

#if defined(RCCL_TEST_CODE_COVERAGE)
#include <dlfcn.h>
extern "C" int __llvm_profile_write_file(void);
#endif

namespace RcclUnitTesting
{

// Sentinel environment variable set by the fork child before execv(). When
// the re-exec'd process finds this variable set, executeAllTests() runs the
// matching test lambda inline (no further fork/exec) then _exit()s with the
// result.
static constexpr const char* kReexecMarkerEnvVar = "RCCL_PIT_REEXEC_TEST";

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
    : name(testName), testLogic(logic), timeout(30), inheritParentEnv(true), numGpus(1)
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

ProcessIsolatedTestRunner::TestConfig& ProcessIsolatedTestRunner::TestConfig::withNumGpus(size_t n)
{
    numGpus = n;
    return *this;
}

// Detect the set of physical GPU device indices available to this process.
// Priority order:
//   1. HIP_VISIBLE_DEVICES — parsed as a comma-separated list of device indices.
//   2. ROCR_VISIBLE_DEVICES — same format (used by some ROCm versions).
//   3. /dev/dri/renderD* node count — each node represents one GPU on ROCm/AMDGPU;
//      produces pool [0, 1, ..., N-1].
//
// Returns an empty vector when no GPUs are detected, which disables GPU slot
// management (tests run without HIP_VISIBLE_DEVICES restrictions).
static std::vector<int> detectGpuPool()
{
    for(const char* envName : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"})
    {
        const char* val = std::getenv(envName);
        if(val && *val)
        {
            std::vector<int>  pool;
            std::istringstream ss(val);
            std::string        token;
            while(std::getline(ss, token, ','))
            {
                try
                {
                    pool.push_back(std::stoi(token));
                }
                catch(...)
                {
                }
            }
            if(!pool.empty())
                return pool;
        }
    }

    int count = 0;
    if(DIR* dir = opendir("/dev/dri"))
    {
        while(struct dirent* e = readdir(dir))
            if(std::strncmp(e->d_name, "renderD", 7) == 0)
                ++count;
        closedir(dir);
    }

    std::vector<int> pool;
    pool.reserve(count);
    for(int i = 0; i < count; ++i)
        pool.push_back(i);
    return pool;
}

// ExecutionOptions implementation
ProcessIsolatedTestRunner::ExecutionOptions::ExecutionOptions()
    : stopOnFirstFailure(false), verboseLogging(true), maxParallelJobs(1)
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
        // Environment was already applied in the fork child before execv().
        // Do NOT call applyEnvironmentVariables() here: for configs with
        // inheritParentEnv=false it would invoke clearenv() a second time,
        // wiping HIP_VISIBLE_DEVICES that the parallel GPU slot manager
        // injected before re-exec.

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
                TEST_INFO(
                    "Test '%s' TIMED OUT after %ld seconds",
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
        TEST_INFO("Test '%s' FAILED with exception: %s", config.name.c_str(), e.what());
        std::cerr << "Exception in test '" << config.name << "': " << e.what() << std::endl;
        fflush(NULL);
        return RCCL_TEST_FAILURE;
    }
    catch(...)
    {
        TEST_INFO("Test '%s' FAILED with unknown exception", config.name.c_str());
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

    // Drain stdout and stderr interleaved via poll() to avoid a deadlock
    // where the child blocks writing one pipe while the parent is blocked
    // reading the other (possible when either pipe buffer fills, ~64 KB).
    struct pollfd pfds[2];
    pfds[0] = {stdoutPipe[0], POLLIN, 0};
    pfds[1] = {stderrPipe[0], POLLIN, 0};
    int openFds = 2;

    while(openFds > 0)
    {
        int ready = poll(pfds, 2, -1 /*block indefinitely*/);
        if(ready < 0)
        {
            if(errno == EINTR) continue;
            break; // unexpected error -- fall through to waitpid
        }
        for(int i = 0; i < 2; ++i)
        {
            if(pfds[i].fd < 0) continue;

            if(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))
            {
                count = read(pfds[i].fd, buffer, sizeof(buffer) - 1);
                if(count > 0)
                {
                    buffer[count] = '\0';
                    (i == 0 ? output.stdoutContent : output.stderrContent)
                        += buffer;
                }
                else
                {
                    // EOF or error -- no more data from this fd.
                    close(pfds[i].fd);
                    pfds[i].fd = -1; // poll ignores negative fds
                    --openFds;
                }
            }
        }
    }

    // Wait for child to exit (blocking).
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

    // Re-exec child entrypoint. Every isolated test is spawned via
    // fork()+execve() of the same binary with --gtest_filter set to the
    // currently running TEST(). The re-exec'd process re-enters the TEST()
    // body, calls RUN_ISOLATED_TESTS again, and lands here. The sentinel env
    // var tells us to run the matching lambda inline (no further fork/exec)
    // and exit with its result.
    if(const char* target = std::getenv(kReexecMarkerEnvVar))
    {
        // Unset so any nested RUN_ISOLATED_TESTS in the lambda doesn't recurse.
        unsetenv(kReexecMarkerEnvVar);
        for(const auto& testConfig : testsToRun)
        {
            if(testConfig.name == target)
            {
                int result = runTestInProcess(testConfig);
                fflush(NULL);
#if defined(RCCL_TEST_CODE_COVERAGE)
                __llvm_profile_write_file();
                using WriteFn = int (*)(void);
                auto libWrite = reinterpret_cast<WriteFn>(
                    dlsym(RTLD_DEFAULT, "rcclCoverageWriteFile"));
                if(libWrite) libWrite();
#endif
                _exit(result);
            }
        }
        std::cerr << "ProcessIsolatedTestRunner: re-exec target '" << target
                  << "' not found in registered tests" << std::endl;
        fflush(NULL);
        _exit(RCCL_TEST_INVALID);
    }

    // Capture the gtest test-info once in the parent — current_test_info() is
    // a single global (not thread-local) and remains valid for the lifetime of
    // this TEST() body, including across background threads.
    const ::testing::TestInfo* gtestInfo
        = ::testing::UnitTest::GetInstance()->current_test_info();

    // Per-test work unit: fork()+execv() the same binary with a gtest filter
    // and the sentinel env var, then drain the child's pipes and wait.
    // Captured as a lambda so it can be called from both the sequential and
    // parallel paths without code duplication.
    //
    // Safe to call from multiple threads simultaneously: env modifications
    // (applyEnvironmentVariables, setenv) happen inside the fork child which
    // is always single-threaded; the parent threads only touch their own pipe
    // file descriptors and the blocking waitpid() for their own child PID.
    struct SpawnOutcome
    {
        TestResult     result;
        CapturedOutput output;
    };

    // Helper: build a comma-separated string from a list of GPU device ids.
    auto formatGpuList = [](const std::vector<int>& ids) -> std::string
    {
        std::string s;
        for(size_t i = 0; i < ids.size(); ++i)
        {
            if(i) s += ',';
            s += std::to_string(ids[i]);
        }
        return s;
    };

    // spawnOne: fork+execv a fresh copy of the binary to run a single isolated
    // test.  `assignedGpus` lists the physical device indices this child is
    // allowed to use; it is injected via HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
    // in the fork child so that concurrent tests never share a GPU.
    // An empty `assignedGpus` means no restriction (sequential mode).
    auto spawnOne = [&](const TestConfig& cfg, const std::vector<int>& assignedGpus)
        -> SpawnOutcome
    {
        auto startTime = std::chrono::steady_clock::now();

        int stdout_fd[2], stderr_fd[2];
        if(!createOutputPipes(stdout_fd, stderr_fd))
        {
            TestResult r;
            r.testName     = cfg.name;
            r.passed       = false;
            r.exitCode     = RCCL_TEST_INVALID;
            r.errorMessage = "Failed to create output pipes";
            return {r, {}};
        }

        // Flush all output before fork to prevent child from inheriting
        // unflushed stdio buffers.
        fflush(NULL);

        pid_t pid = fork();

        if(pid == 0)
        {
            // Always re-exec: replace this child image with a fresh copy of
            // the binary. execv() discards all counters before any coverage
            // data is touched; flushing is handled in the re-exec entrypoint.
            redirectOutputToPipes(stdout_fd, stderr_fd);
            applyEnvironmentVariables(cfg);

            // Restrict GPU visibility to the assigned subset so this child
            // cannot accidentally use a GPU that a sibling test is using.
            // Both env vars are set for compatibility across ROCm versions.
            if(!assignedGpus.empty())
            {
                std::string ids = formatGpuList(assignedGpus);
                setenv("HIP_VISIBLE_DEVICES", ids.c_str(), 1 /*overwrite*/);
                setenv("ROCR_VISIBLE_DEVICES", ids.c_str(), 1);
            }

            setenv(kReexecMarkerEnvVar, cfg.name.c_str(), 1);

            if(!gtestInfo)
            {
                std::cerr << "ProcessIsolatedTestRunner: executeAllTests() must be "
                             "called from within a gtest TEST() body; "
                             "current_test_info() is null."
                          << std::endl;
                fflush(NULL);
                _exit(RCCL_TEST_INVALID);
            }
            std::string filterArg = std::string("--gtest_filter=")
                                    + gtestInfo->test_suite_name() + "." + gtestInfo->name();
            // Disable ANSI color in the child: its output is captured via pipe
            // and replayed by the parent, so escape sequences are unwanted.
            char  argv0[]    = "/proc/self/exe";
            char  colorArg[] = "--gtest_color=no";
            char* argv[]     = {argv0, filterArg.data(), colorArg, nullptr};
            execv("/proc/self/exe", argv);
            std::cerr << "ProcessIsolatedTestRunner: execv(/proc/self/exe) failed: "
                      << strerror(errno) << std::endl;
            fflush(NULL);
            _exit(RCCL_TEST_INVALID);
        }
        else if(pid < 0)
        {
            close(stdout_fd[0]);
            close(stdout_fd[1]);
            close(stderr_fd[0]);
            close(stderr_fd[1]);
            TestResult r;
            r.testName     = cfg.name;
            r.passed       = false;
            r.exitCode     = RCCL_TEST_INVALID;
            r.processId    = RCCL_TEST_INVALID;
            r.duration     = std::chrono::milliseconds(0);
            r.errorMessage = "Failed to fork process";
            TEST_INFO("Failed to fork process for test '%s'", cfg.name.c_str());
            return {r, {}};
        }

        // Parent: log launch, drain pipes, wait for child.
        {
            std::string extras;
            if(!assignedGpus.empty())
                extras += " GPUs: " + formatGpuList(assignedGpus);
            if(!cfg.environmentVariables.empty())
            {
                std::string envVars;
                for(const auto& [name, value] : cfg.environmentVariables)
                {
                    if(!envVars.empty())
                        envVars += ", ";
                    envVars += name + "=" + value;
                }
                extras += " env: " + envVars;
            }
            if(!extras.empty())
                TEST_INFO(
                    "Running isolated test '%s' (PID: %d) with%s",
                    cfg.name.c_str(), pid, extras.c_str()
                );
            else
                TEST_INFO("Running isolated test '%s' (PID: %d)", cfg.name.c_str(), pid);
        }

        int            status;
        CapturedOutput output = captureProcessOutput(stdout_fd, stderr_fd, pid, &status);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime
        );

        TestResult result;
        result.testName  = cfg.name;
        result.processId = pid;
        result.duration  = duration;

        if(WIFEXITED(status))
        {
            int exitCode    = WEXITSTATUS(status);
            result.exitCode = exitCode;
            result.passed   = (exitCode == RCCL_TEST_SUCCESS);
            result.skipped  = (exitCode == RCCL_TEST_SKIPPED);

            if(exitCode == RCCL_TEST_SUCCESS)
            {
                TEST_INFO("Test '%s' PASSED (%ld ms)", cfg.name.c_str(), duration.count());
            }
            else if(exitCode == RCCL_TEST_TIMEOUT)
            {
                TEST_INFO(
                    "Test '%s' (PID: %d) TIMED OUT after %ld ms",
                    cfg.name.c_str(), pid, duration.count()
                );
                result.errorMessage = "Test timed out";
            }
            else if(exitCode == RCCL_TEST_SKIPPED)
            {
                TEST_INFO(
                    "Test '%s' (PID: %d) SKIPPED in %ld ms",
                    cfg.name.c_str(), pid, duration.count()
                );
                result.errorMessage = "Test skipped";
            }
            else
            {
                TEST_INFO(
                    "Test '%s' (PID: %d) FAILED with exit code %d after %ld ms",
                    cfg.name.c_str(), pid, exitCode, duration.count()
                );
                result.errorMessage
                    = "Test failed with exit code " + std::to_string(exitCode);
            }
        }
        else if(WIFSIGNALED(status))
        {
            int signal = WTERMSIG(status);
            if(output.stdoutContent.find("PASSED") != std::string::npos)
            {
                result.passed   = true;
                result.exitCode = RCCL_TEST_SUCCESS;
                TEST_INFO("Test '%s' PASSED (%ld ms)", cfg.name.c_str(), duration.count());
            }
            else
            {
                result.passed       = false;
                result.exitCode     = -signal;
                result.errorMessage = "Terminated by signal " + std::to_string(signal);
                TEST_INFO(
                    "Test '%s' (PID: %d) terminated by signal %d after %ld ms",
                    cfg.name.c_str(), pid, signal, duration.count()
                );
            }
        }
        else
        {
            result.passed       = false;
            result.exitCode     = RCCL_TEST_INVALID;
            result.errorMessage = "Failed to wait for process";
        }

        return {result, output};
    };

    // ── Determine parallelism ──────────────────────────────────────────────
    // When maxParallelJobs is 0, default to the GPU pool size so threads don't
    // pile up waiting for slots on GPU-test suites; fall back to
    // hardware_concurrency() when no GPU pool exists.
    auto computeDefaultParallelism = [&]() -> size_t
    {
        const std::vector<int> pool
            = options.gpuPool.empty() ? detectGpuPool() : options.gpuPool;
        return pool.empty() ? static_cast<size_t>(std::thread::hardware_concurrency())
                            : pool.size();
    };

    const size_t parallelism
        = (options.maxParallelJobs == 0) ? computeDefaultParallelism()
                                         : options.maxParallelJobs;

    if(parallelism <= 1)
    {
        for(const auto& testConfig : testsToRun)
        {
            auto [result, output] = spawnOne(testConfig, {});
            displayCapturedOutput(output, testConfig.name);
            recordTestResult(result);

            if(options.stopOnFirstFailure && !result.passed && !result.skipped)
                break;
        }
    }
    else
    {
        // Bounded sliding window: up to `parallelism` children run at once.
        // GPU tests receive a non-overlapping slice of the pool via
        // HIP_VISIBLE_DEVICES; concurrency and GPU slot availability are
        // checked atomically to avoid TOCTOU races.
        std::vector<int> gpuPool
            = options.gpuPool.empty() ? detectGpuPool() : options.gpuPool;

        std::vector<bool> gpuSlotInUse(gpuPool.size(), false);

        const bool gpuMgmtEnabled = !gpuPool.empty();

        if(gpuMgmtEnabled)
            TEST_INFO(
                "GPU slot manager: pool = [%s] (%zu device(s))",
                formatGpuList(gpuPool).c_str(), gpuPool.size()
            );

        // Caller must hold cvMtx for all three helpers below.
        auto gpuAcquireLocked = [&](size_t n) -> std::vector<int>
        {
            std::vector<int> assigned;
            assigned.reserve(n);
            for(size_t i = 0; i < gpuPool.size() && assigned.size() < n; ++i)
            {
                if(!gpuSlotInUse[i])
                {
                    gpuSlotInUse[i] = true;
                    assigned.push_back(gpuPool[i]);
                }
            }
            return assigned;
        };

        auto gpuReleaseLocked = [&](const std::vector<int>& physIds)
        {
            for(int id : physIds)
                for(size_t i = 0; i < gpuPool.size(); ++i)
                    if(gpuPool[i] == id)
                    {
                        gpuSlotInUse[i] = false;
                        break;
                    }
        };

        auto gpuFreeLocked = [&]() -> size_t
        {
            return static_cast<size_t>(
                std::count(gpuSlotInUse.begin(), gpuSlotInUse.end(), false)
            );
        };

        std::mutex              cvMtx;
        std::condition_variable cv;
        size_t                  active = 0;
        std::atomic<bool>       anyFailed{false};
        const bool              stopOnFirst = options.stopOnFirstFailure;

        using Outcome = std::pair<TestResult, CapturedOutput>;
        std::vector<std::future<Outcome>> futures;
        futures.reserve(testsToRun.size());

        for(const auto& testConfig : testsToRun)
        {
            // 0 = CPU-only (no slot); > 0 = GPU slots needed.
            const size_t need = (gpuMgmtEnabled && testConfig.numGpus > 0)
                                    ? testConfig.numGpus
                                    : 0;
            // Clamp to pool size: over-requesting runs exclusively so no
            // sibling can hold a subset and cause resource contention.
            const size_t effectiveNeed = std::min(need, gpuPool.size());

            if(need > 0 && need > gpuPool.size())
            {
                TEST_INFO(
                    "WARNING: test '%s' requests %zu GPU(s) but pool has only %zu — "
                    "assigning the entire pool and running exclusively",
                    testConfig.name.c_str(), need, gpuPool.size()
                );
            }

            std::vector<int> assignedGpus;
            {
                std::unique_lock<std::mutex> lk(cvMtx);
                cv.wait(lk, [&]
                {
                    if(stopOnFirst && anyFailed.load()) return true;
                    if(active >= parallelism) return false;
                    if(effectiveNeed > 0 && gpuFreeLocked() < effectiveNeed)
                        return false;
                    return true;
                });

                if(stopOnFirst && anyFailed.load())
                    break;

                ++active;
                if(effectiveNeed > 0)
                    assignedGpus = gpuAcquireLocked(effectiveNeed);
            }

            futures.push_back(std::async(
                std::launch::async,
                [testConfig, assignedGpus, stopOnFirst, &spawnOne, &active, &cv, &cvMtx,
                 &gpuReleaseLocked, &anyFailed]() -> Outcome
                {
                    auto so = spawnOne(testConfig, assignedGpus);
                    {
                        std::lock_guard<std::mutex> lk(cvMtx);
                        --active;
                        gpuReleaseLocked(assignedGpus);
                        if(stopOnFirst && !so.result.passed && !so.result.skipped)
                            anyFailed.store(true);
                    }
                    cv.notify_all();
                    return {std::move(so.result), std::move(so.output)};
                }
            ));
        }

        // Drain futures in registration order (including in-flight ones after a break).
        for(auto& future : futures)
        {
            auto [result, output] = future.get();
            displayCapturedOutput(output, result.testName);
            recordTestResult(result);
        }
    }

    bool result = generateReport(options, testsToRun.size());
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
bool ProcessIsolatedTestRunner::generateReport(
    const ExecutionOptions& options, size_t totalRegistered
)
{
    int                       totalTests   = 0;
    int                       passedTests  = 0;
    int                       failedTests  = 0;
    int                       skippedTests = 0;
    std::chrono::milliseconds totalDuration{0};

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        totalTests = static_cast<int>(testResults_.size());

        for(const auto& result : testResults_)
        {
            if(result.skipped)
                skippedTests++;
            else if(result.passed)
                passedTests++;
            else
                failedTests++;

            totalDuration += result.duration;
        }
    }

    const int notRunTests = static_cast<int>(totalRegistered) - totalTests;

    if(failedTests > 0 || totalTests > 1 || notRunTests > 0)
    {
        if(notRunTests > 0)
            TEST_INFO(
                "Process-Isolated Tests: %d passed, %d failed, %d skipped, "
                "%d not run (stopped on first failure) (%ld ms total)",
                passedTests,
                failedTests,
                skippedTests,
                notRunTests,
                totalDuration.count()
            );
        else
            TEST_INFO(
                "Process-Isolated Tests: %d passed, %d failed, %d skipped (%ld ms total)",
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
                    TEST_INFO(
                        "  Failed: %s - %s",
                        result.testName.c_str(),
                        result.errorMessage.c_str()
                    );
                }
            }
        }
    }

    return failedTests == 0 && notRunTests == 0;
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
