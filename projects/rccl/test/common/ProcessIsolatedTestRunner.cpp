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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

#include "ErrCode.hpp"

#if defined(RCCL_TEST_CODE_COVERAGE)
#include <dlfcn.h>
extern "C" int __llvm_profile_write_file(void);
#endif

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

// Grace period given to a timed-out child after SIGTERM before SIGKILL.
static constexpr int kSigtermGraceSeconds = 2;

// Maximum poll(2) slice when draining child output with an active deadline.
// Caps the wait so the deadline is re-checked regularly even if output is slow.
static constexpr long long kPollSliceMs = 5000LL;

// Number of pipe file descriptors polled per child process (stdout + stderr).
static constexpr int kNumPipeFds = 2;

// POSIX waitpid(2): exit code is stored in bits [15:8] of the status word.
static constexpr int kWaitStatusExitShift = 8;

// Read buffer size for draining child stdout/stderr pipes.
static constexpr int kPipeReadBufferSize = 4096;

// poll(2) timeout sentinel: block indefinitely until an fd becomes ready.
static constexpr int kPollBlockIndefinitely = -1;

// pollfd.fd sentinel: poll(2) ignores entries with a negative fd.
static constexpr int kDisabledFd = -1;

// Default test timeout (seconds) if none is specified in TestConfig.
static constexpr int kDefaultTestTimeoutSeconds = 30;

// Unset process ID — used before a child is forked or when fork fails.
static constexpr pid_t kInvalidPid = -1;

// maxParallelJobs value that selects single-child sequential execution.
static constexpr size_t kSequentialExecution = 1;

// Define static members
std::mutex                                         ProcessIsolatedTestRunner::testConfigsMutex_;
std::vector<ProcessIsolatedTestRunner::TestConfig> ProcessIsolatedTestRunner::testConfigs_;
std::mutex                                         ProcessIsolatedTestRunner::resultsMutex_;
std::vector<ProcessIsolatedTestRunner::TestResult> ProcessIsolatedTestRunner::testResults_;

// TestResult implementation
ProcessIsolatedTestRunner::TestResult::TestResult()
    : passed(false), skipped(false), exitCode(RCCL_TEST_INVALID), processId(kInvalidPid), duration(0)
{}

// TestConfig implementation
ProcessIsolatedTestRunner::TestConfig::TestConfig(
    const std::string& testName, std::function<void()> logic
)
    : name(testName), testLogic(logic), timeout(kDefaultTestTimeoutSeconds), inheritParentEnv(true), numGpus(0)
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
    ProcessIsolatedTestRunner::TestConfig::withCleanEnvironment(bool cleanEnv)
{
    inheritParentEnv = !cleanEnv;
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

// Returns GPU indices from HIP_VISIBLE_DEVICES, or empty if unset/invalid.
static std::vector<int> gpuPoolFromEnv()
{
    const char* val = std::getenv("HIP_VISIBLE_DEVICES");
    if(!val || !*val)
        return {};

    std::vector<int>   pool;
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
    return pool;
}

// Returns GPU indices from KFD topology sysfs, or empty if unavailable.
// KFD topology nodes with gpu_id == 0 represent CPU or bridge nodes and are
// skipped; only nodes with a non-zero gpu_id correspond to physical GPUs.
// Uses sysfs rather than opening a KFD device file to avoid inheriting GPU
// file descriptors in child processes.
static std::vector<int> gpuPoolFromKfd()
{
    namespace fs = std::filesystem;
    const fs::path nodesDir{"/sys/class/kfd/kfd/topology/nodes"};

    int kfdCount = 0;
    std::error_code ec;
    for(const auto& entry : fs::directory_iterator(nodesDir, ec))
    {
        if(std::ifstream f{entry.path() / "gpu_id"})
        {
            unsigned gpuId = 0;
            if(f >> gpuId && gpuId != 0)
                ++kfdCount;
        }
    }
    if(kfdCount == 0)
        return {};

    std::vector<int> pool(static_cast<size_t>(kfdCount));
    std::iota(pool.begin(), pool.end(), 0);
    return pool;
}

// Detects available GPU indices: first from HIP_VISIBLE_DEVICES, then from
// KFD sysfs. Returns empty if neither source yields a pool.
static std::vector<int> detectGpuPool()
{
    auto pool = gpuPoolFromEnv();
    if(pool.empty())
        pool = gpuPoolFromKfd();
    return pool;
}

// Tracks which physical GPU device indices from a fixed pool are in use.
// acquire() and release() must be called with the external mutex (cvMtx) held.
class GpuSlotManager
{
public:
    explicit GpuSlotManager(std::vector<int> pool)
        : pool_(std::move(pool)), inUse_(pool_.size(), false)
    {}

    bool   empty() const { return pool_.empty(); }
    size_t size()  const { return pool_.size(); }
    const std::vector<int>& pool() const { return pool_; }

    // Number of currently free slots.  Must be called with the external mutex held.
    size_t freeSlots() const
    {
        return static_cast<size_t>(
            std::count(inUse_.begin(), inUse_.end(), false));
    }

    // Assigns `n` free slots and returns their physical device IDs.
    // Precondition: n <= size(); must be called with the external mutex held.
    std::vector<int> acquire(size_t n)
    {
        std::vector<int> assigned;
        assigned.reserve(n);
        for(size_t i = 0; i < pool_.size() && assigned.size() < n; ++i)
        {
            if(!inUse_[i])
            {
                inUse_[i] = true;
                assigned.push_back(pool_[i]);
            }
        }
        return assigned;
    }

    // Return previously acquired slots back to the pool.
    // Must be called with the external mutex held.
    void release(const std::vector<int>& physIds)
    {
        for(int id : physIds)
            for(size_t i = 0; i < pool_.size(); ++i)
                if(pool_[i] == id)
                {
                    inUse_[i] = false;
                    break;
                }
    }

    // Format a list of device indices as a comma-separated string.
    static std::string formatList(const std::vector<int>& ids)
    {
        std::string s;
        for(size_t i = 0; i < ids.size(); ++i)
        {
            if(i) s += ',';
            s += std::to_string(ids[i]);
        }
        return s;
    }

private:
    std::vector<int>  pool_;
    std::vector<bool> inUse_;
};

// ExecutionOptions implementation
ProcessIsolatedTestRunner::ExecutionOptions::ExecutionOptions()
    : stopOnFirstFailure(false), verboseLogging(true), maxParallelJobs(kSequentialExecution)
{}

// Apply environment variables to current process
void ProcessIsolatedTestRunner::applyEnvironmentVariables(const TestConfig& config)
{
    for(const auto& varName : config.clearEnvVars)
        unsetenv(varName.c_str());

    if(!config.inheritParentEnv)
        if(clearenv() != 0)
            std::cerr << "Warning: Failed to clear environment variables\n";

    for(const auto& [name, value] : config.environmentVariables)
        setenv(name.c_str(), value.c_str(), 1);
}

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
        const ::testing::UnitTest* unitTest            = ::testing::UnitTest::GetInstance();
        const size_t               initialFailureCount = unitTest->failed_test_count();
        const size_t               initialSkippedCount = unitTest->skipped_test_count();

        // Package the test as a future so wait_for() can enforce the deadline
        // without a busy-wait loop.
        std::packaged_task<int()> task(
            [&]() -> int
            {
                config.testLogic();
                const bool passed  = (unitTest->failed_test_count()  == initialFailureCount);
                const bool skipped = (unitTest->skipped_test_count()  >  initialSkippedCount);
                return skipped ? RCCL_TEST_SKIPPED
                     : passed  ? RCCL_TEST_SUCCESS
                               : RCCL_TEST_FAILURE;
            }
        );
        auto fut = task.get_future();
        std::thread testThread(std::move(task));

        if(fut.wait_for(config.timeout) == std::future_status::timeout)
        {
            TEST_INFO(
                "Test '%s' TIMED OUT after %ld seconds",
                config.name.c_str(),
                config.timeout.count()
            );
            fflush(NULL);
            testThread.detach();
            return RCCL_TEST_TIMEOUT;
        }

        testThread.join();
        fflush(NULL);
        return fut.get(); // re-throws if the task threw
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
    // Null bytes are silently truncated by setenv()/getenv(), which would
    // cause the re-exec child to fail to find its target test.
    if(config.name.find('\0') != std::string::npos)
    {
        std::cerr << "ProcessIsolatedTestRunner: test name contains a null "
                     "byte and cannot be registered: '" << config.name
                  << "'\n";
        return;
    }

    std::lock_guard<std::mutex> lock(testConfigsMutex_);

    // Duplicate names cause the re-exec child to always match the first
    // registration and silently skip the second.
    for(const auto& existing : testConfigs_)
    {
        if(existing.name == config.name)
        {
            std::cerr << "ProcessIsolatedTestRunner: duplicate test name '"
                      << config.name << "' — second registration ignored.\n";
            return;
        }
    }

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

// Reads stdout/stderr pipes until the child exits or the deadline is reached.
// On timeout: SIGTERM, then SIGKILL after 2 s; *status set to RCCL_TEST_TIMEOUT.
ProcessIsolatedTestRunner::CapturedOutput ProcessIsolatedTestRunner::captureProcessOutput(
    int stdoutPipe[2], int stderrPipe[2], pid_t pid, int* status,
    std::chrono::seconds timeout
)
{
    // Close write ends of pipes in parent process (not needed)
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    CapturedOutput output;
    char           buffer[kPipeReadBufferSize];
    ssize_t        nbytes;

    using Clock    = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    const bool    hasTimeout = (timeout.count() > 0);
    const TimePoint deadline = hasTimeout ? Clock::now() + timeout
                                          : TimePoint::max();

    // Drain stdout and stderr interleaved via poll() to avoid deadlock when
    // the child fills one pipe buffer (~64 KB) while the parent reads the other.
    struct pollfd pfds[kNumPipeFds];
    pfds[0] = {stdoutPipe[0], POLLIN, 0};
    pfds[1] = {stderrPipe[0], POLLIN, 0};
    int openFds = kNumPipeFds;

    bool timedOut = false;

    while(openFds > 0)
    {
        int pollMs = kPollBlockIndefinitely;
        if(hasTimeout)
        {
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - Clock::now()
            );
            if(remaining.count() <= 0)
            {
                timedOut = true;
                break;
            }
            // Cap to kPollSliceMs so we re-check the deadline regularly.
            auto capMs = std::min<long long>(remaining.count(), kPollSliceMs);
            pollMs     = static_cast<int>(capMs);
        }

        int ready = poll(pfds, kNumPipeFds, pollMs);
        if(ready < 0)
        {
            if(errno == EINTR) continue;
            break; // unexpected error -- fall through to waitpid
        }
        if(ready == 0)
        {
            // poll timed out — re-check deadline next iteration
            continue;
        }
        for(int i = 0; i < kNumPipeFds; ++i)
        {
            if(pfds[i].fd < 0) continue;

            if(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))
            {
                nbytes = read(pfds[i].fd, buffer, sizeof(buffer) - 1);
                if(nbytes > 0)
                {
                    buffer[nbytes] = '\0';
                    (i == 0 ? output.stdoutContent : output.stderrContent)
                        += buffer;
                }
                else
                {
                    // EOF or error -- no more data from this fd.
                    close(pfds[i].fd);
                    pfds[i].fd = kDisabledFd;
                    --openFds;
                }
            }
        }
    }

    if(timedOut)
    {
        // Give the child a chance to exit cleanly, then force-kill it.
        kill(pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::seconds(kSigtermGraceSeconds));
        kill(pid, SIGKILL);

        // Drain any last output flushed before death.
        for(int i = 0; i < kNumPipeFds; ++i)
        {
            if(pfds[i].fd < 0) continue;
            // Non-blocking drain
            fcntl(pfds[i].fd, F_SETFL, O_NONBLOCK);
            while((nbytes = read(pfds[i].fd, buffer, sizeof(buffer) - 1)) > 0)
            {
                buffer[nbytes] = '\0';
                (i == 0 ? output.stdoutContent : output.stderrContent) += buffer;
            }
            close(pfds[i].fd);
        }

        waitpid(pid, status, 0);

        // Synthesise a waitpid(2) status as if the child called exit(RCCL_TEST_TIMEOUT).
        // POSIX encodes the exit code in bits [15:8] of the status word, so
        // WIFEXITED(*status) is true and WEXITSTATUS(*status) == RCCL_TEST_TIMEOUT.
        *status = RCCL_TEST_TIMEOUT << kWaitStatusExitShift;
        return output;
    }

    // Normal path: wait for child to exit.
    waitpid(pid, status, 0);

    return output;
}

// Helper method: Display captured output
void ProcessIsolatedTestRunner::displayCapturedOutput(
    const CapturedOutput& output, const std::string& /*testName*/
)
{
    auto flush = [](std::ostream& os, const std::string& s) {
        if(s.empty()) return;
        os << s;
        if(s.back() != '\n') os << '\n';
    };
    flush(std::cout, output.stdoutContent);
    flush(std::cerr, output.stderrContent);
}

// Re-exec child entrypoint: if kReexecMarkerEnvVar is set, runs the named test
// and _exit()s with the result. Returns normally only in the original parent.
void ProcessIsolatedTestRunner::handleReexecEntrypoint(
    const std::vector<TestConfig>& tests)
{
    const char* target = std::getenv(kReexecMarkerEnvVar);
    if(!target)
        return;

    // Unset so any nested RUN_ISOLATED_TESTS in the lambda doesn't recurse.
    unsetenv(kReexecMarkerEnvVar);

    for(const auto& testConfig : tests)
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

void ProcessIsolatedTestRunner::runSequential(
    const std::vector<TestConfig>& tests,
    const ExecutionOptions&        opts,
    const SpawnFn&                 spawnFn)
{
    for(const auto& testConfig : tests)
    {
        auto outcome = spawnFn(testConfig, {});
        displayCapturedOutput(outcome.output, testConfig.name);
        recordTestResult(outcome.result);

        if(opts.stopOnFirstFailure && !outcome.result.passed && !outcome.result.skipped)
            break;
    }
}

void ProcessIsolatedTestRunner::runParallel(
    const std::vector<TestConfig>& tests,
    const ExecutionOptions&        opts,
    size_t                         parallelism,
    const std::vector<int>&        gpuPool,
    const SpawnFn&                 spawnFn)
{
    // Bounded sliding window: up to `parallelism` children run at once.
    // GPU slot availability and active count are checked atomically.
    GpuSlotManager slots(gpuPool);

    if(!slots.empty())
        TEST_INFO(
            "GPU slot manager: pool = [%s] (%zu device(s))",
            GpuSlotManager::formatList(slots.pool()).c_str(), slots.size()
        );

    std::mutex              cvMtx;
    std::condition_variable cv;
    size_t                  active = 0;
    std::atomic<bool>       anyFailed{false};
    const bool              stopOnFirst = opts.stopOnFirstFailure;

    using OutcomePair = std::pair<TestResult, CapturedOutput>;
    std::vector<std::future<OutcomePair>> futures;
    futures.reserve(tests.size());

    for(const auto& testConfig : tests)
    {
        // 0 = CPU-only (no slot); > 0 = GPU slots needed.
        const size_t need = (!slots.empty() && testConfig.numGpus > 0)
                                ? testConfig.numGpus
                                : 0;
        // Clamp to pool size: over-requesting runs exclusively so no
        // sibling can hold a subset and cause resource contention.
        const size_t effectiveNeed = std::min(need, slots.size());

        if(need > 0 && need > slots.size())
        {
            TEST_INFO(
                "WARNING: test '%s' requests %zu GPU(s) but pool has only %zu — "
                "assigning the entire pool and running exclusively",
                testConfig.name.c_str(), need, slots.size()
            );
        }

        std::vector<int> assignedGpus;
        {
            std::unique_lock<std::mutex> lk(cvMtx);
            cv.wait(lk, [&]
            {
                if(stopOnFirst && anyFailed.load()) return true;
                if(active >= parallelism) return false;
                if(effectiveNeed > 0 && slots.freeSlots() < effectiveNeed)
                    return false;
                return true;
            });

            if(stopOnFirst && anyFailed.load())
                break;

            ++active;
            if(effectiveNeed > 0)
                assignedGpus = slots.acquire(effectiveNeed);
        }

        futures.push_back(std::async(
            std::launch::async,
            [testConfig, assignedGpus, stopOnFirst, &spawnFn, &active, &cv, &cvMtx,
             &slots, &anyFailed]() -> OutcomePair
            {
                auto outcome = spawnFn(testConfig, assignedGpus);
                {
                    std::lock_guard<std::mutex> lk(cvMtx);
                    --active;
                    slots.release(assignedGpus);
                    if(stopOnFirst && !outcome.result.passed && !outcome.result.skipped)
                        anyFailed.store(true);
                }
                cv.notify_all();
                return {std::move(outcome.result), std::move(outcome.output)};
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

    // If this process is a re-exec child, handleReexecEntrypoint() runs the
    // target test and calls _exit() — it returns only in the parent.
    handleReexecEntrypoint(testsToRun);

    // Capture gtest test-info once in the parent; current_test_info() is a
    // single global valid for the lifetime of this TEST() body.
    const ::testing::TestInfo* gtestInfo
        = ::testing::UnitTest::GetInstance()->current_test_info();
    if(!gtestInfo)
    {
        std::cerr << "ProcessIsolatedTestRunner: executeAllTests() must be "
                     "called from within a gtest TEST() body; "
                     "current_test_info() is null." << std::endl;
        return false;
    }

    // spawnOne: fork+execv a fresh binary copy to run one isolated test.
    // Thread-safe: env changes happen only in the fork child; parents own their pipe fds.
    auto spawnOne = [&](const TestConfig& cfg, const std::vector<int>& assignedGpus)
        -> SpawnOutcome
    {
        auto makeError = [&](const char* msg) -> SpawnOutcome {
            TestResult r;
            r.testName     = cfg.name;
            r.passed       = false;
            r.exitCode     = RCCL_TEST_INVALID;
            r.errorMessage = msg;
            return {r, {}};
        };

        auto startTime = std::chrono::steady_clock::now();

        int stdout_fd[2], stderr_fd[2];
        if(!createOutputPipes(stdout_fd, stderr_fd))
            return makeError("Failed to create output pipes");

        // Flush all output before fork to prevent child from inheriting
        // unflushed stdio buffers.
        fflush(NULL);

        pid_t pid = fork();

        if(pid == 0)
        {
            // Always re-exec to get a fresh binary image; execv() discards
            // all counters before coverage data is touched.
            redirectOutputToPipes(stdout_fd, stderr_fd);
            applyEnvironmentVariables(cfg);

            // Restrict GPU visibility to the assigned subset so this child
            // cannot accidentally share a GPU with a sibling test.
            if(!assignedGpus.empty())
            {
                std::string ids = GpuSlotManager::formatList(assignedGpus);

                // Warn when the slot manager overrides a test-specified value;
                // the override is intentional for isolation.
                const char* existingHVD = std::getenv("HIP_VISIBLE_DEVICES");
                if(existingHVD && *existingHVD)
                    std::cerr
                        << "ProcessIsolatedTestRunner: GPU slot manager overrides "
                           "test-specified HIP_VISIBLE_DEVICES='" << existingHVD
                        << "' with '" << ids
                        << "' for parallel isolation\n";

                setenv("HIP_VISIBLE_DEVICES", ids.c_str(), 1 /*overwrite*/);
            }

            // Give each child its own profraw file (%p=PID, %m=module hash).
            // overwrite=0 preserves any LLVM_PROFILE_FILE already set by CI.
#if defined(RCCL_TEST_CODE_COVERAGE)
            setenv("LLVM_PROFILE_FILE", "rccl_tests_%p_%m.profraw", 0 /*no overwrite*/);
#endif

            setenv(kReexecMarkerEnvVar, cfg.name.c_str(), 1);

            // gtestInfo was validated non-null in the parent before any
            // fork; no need to re-check here.
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
            close(stdout_fd[0]); close(stdout_fd[1]);
            close(stderr_fd[0]); close(stderr_fd[1]);
            TEST_INFO("Failed to fork process for test '%s'", cfg.name.c_str());
            return makeError("Failed to fork process");
        }

        // Parent: log launch, drain pipes, wait for child.
        {
            std::string extras;
            if(!assignedGpus.empty())
                extras += " GPUs: " + GpuSlotManager::formatList(assignedGpus);
            for(const auto& [name, value] : cfg.environmentVariables)
                extras += (extras.empty() ? " env: " : ", ") + name + "=" + value;

            if(!extras.empty())
                TEST_INFO("Running isolated test '%s' (PID: %d) with%s",
                          cfg.name.c_str(), pid, extras.c_str());
            else
                TEST_INFO("Running isolated test '%s' (PID: %d)", cfg.name.c_str(), pid);
        }

        int            status;
        CapturedOutput output = captureProcessOutput(stdout_fd, stderr_fd, pid, &status,
                                                     cfg.timeout);

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
            // With re-exec the child always _exit()s after completion, so a
            // signal indicates a genuine crash or external kill (OOM, SIGSEGV).
            int signal = WTERMSIG(status);
            result.passed       = false;
            result.exitCode     = -signal;
            result.errorMessage = "Terminated by signal " + std::to_string(signal);
            TEST_INFO(
                "Test '%s' (PID: %d) terminated by signal %d after %ld ms",
                cfg.name.c_str(), pid, signal, duration.count()
            );
        }
        else
        {
            result.passed       = false;
            result.exitCode     = RCCL_TEST_INVALID;
            result.errorMessage = "Failed to wait for process";
        }

        return {result, output};
    };

    // Detect the GPU pool once so the parallelism calculation and slot
    // manager always operate on the same snapshot.
    const std::vector<int> detectedPool
        = options.gpuPool.empty() ? detectGpuPool() : options.gpuPool;

    // When maxParallelJobs is kAutoParallelism, default to pool size (or
    // hardware_concurrency() for CPU-only suites) to avoid threads piling up
    // waiting for GPU slots.
    const size_t parallelism
        = (options.maxParallelJobs == ExecutionOptions::kAutoParallelism)
              ? (detectedPool.empty()
                     ? static_cast<size_t>(std::thread::hardware_concurrency())
                     : detectedPool.size())
              : options.maxParallelJobs;

    // A GPU pool is only required when running in parallel with tests that
    // actually request GPU slots. Sequential runs and CPU-only suites proceed
    // without one; kAutoParallelism falls back to hardware_concurrency().
    const bool anyTestNeedsGpu = std::any_of(
        testsToRun.begin(), testsToRun.end(),
        [](const TestConfig& c) { return c.numGpus > TestConfig::kCpuOnly; }
    );
    if(detectedPool.empty() && parallelism > kSequentialExecution && anyTestNeedsGpu)
    {
        TEST_INFO("Could not determine GPU pool for parallel GPU tests.");
        TEST_INFO("  Tried: HIP_VISIBLE_DEVICES (not set or empty),");
        TEST_INFO("         /sys/class/kfd/kfd/topology/nodes (not found or no GPU nodes).");
        TEST_INFO("  Fix: set HIP_VISIBLE_DEVICES=0,1,...  or ensure the KFD sysfs is mounted.");
        ADD_FAILURE() << "Set HIP_VISIBLE_DEVICES or ensure /sys/class/kfd is mounted.";
        return false;
    }

    SpawnFn spawn = spawnOne;
    if(parallelism <= kSequentialExecution)
        runSequential(testsToRun, options, spawn);
    else
        runParallel(testsToRun, options, parallelism, detectedPool, spawn);

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
    // Collect failure details inside the lock so we don't need a second pass.
    std::vector<std::pair<std::string, std::string>> failureDetails;

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        totalTests = static_cast<int>(testResults_.size());
        for(const auto& result : testResults_)
        {
            if(result.skipped)       ++skippedTests;
            else if(result.passed)   ++passedTests;
            else
            {
                ++failedTests;
                failureDetails.emplace_back(result.testName, result.errorMessage);
            }
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
                passedTests, failedTests, skippedTests, notRunTests, totalDuration.count()
            );
        else
            TEST_INFO(
                "Process-Isolated Tests: %d passed, %d failed, %d skipped (%ld ms total)",
                passedTests, failedTests, skippedTests, totalDuration.count()
            );

        for(const auto& [name, msg] : failureDetails)
            TEST_INFO("  Failed: %s - %s", name.c_str(), msg.c_str());
    }

    // notRunTests is shown in the summary but is not itself a failure;
    // the actual failing test drives the false return.
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

    {
        std::lock_guard<std::mutex> lock(testConfigsMutex_);
        registeredCount = testConfigs_.size();
        testConfigs_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        executedCount = testResults_.size();
        testResults_.clear();
    }

    if(registeredCount > 0 && executedCount < registeredCount)
    {
        std::cerr << "\n WARNING: ProcessIsolatedTestRunner::clear() called with "
                  << (registeredCount - executedCount) << " unexecuted test(s)!\n"
                  << "   Registered: " << registeredCount << " test(s)\n"
                  << "   Executed:   " << executedCount << " test(s)\n"
                  << "   Did you forget to call executeAllTests()?\n"
                  << std::endl;
    }
}

// Get number of registered tests
size_t ProcessIsolatedTestRunner::getTestCount()
{
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    return testConfigs_.size();
}

} // namespace RcclUnitTesting
