/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "GPUScheduler.hpp"
#include "ErrCode.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <nccl.h>
#include <gtest/gtest.h>

namespace RcclUnitTesting
{

GPUScheduler::GPUScheduler(const GPUSchedulingConfig& config)
    : config_(config)
    , totalSubmitted_(0)
    , totalCompleted_(0)
    , totalFailed_(0)
    , nextJobId_(0)
    , nextUniqueIdIndex_(0)
    , stopIdGeneration_(false)
    , targetPoolSize_(0)
{
    // Initialize GPU tracking
    gpuInUse_.resize(config_.totalGPUs, false);
    gpuBusyTime_.resize(config_.totalGPUs, std::chrono::milliseconds(0));
    schedulerStartTime_ = std::chrono::steady_clock::now();

    if (config_.verboseLogging)
    {
        INFO("GPUScheduler initialized with %d GPUs, max %d concurrent tests\n",
             config_.totalGPUs, config_.maxConcurrentTests);
    }
}

GPUScheduler::~GPUScheduler()
{
    // Stop background ID generation if running
    stopAsyncIdGeneration();

    // Wait for any remaining tests
    while (!runningTests_.empty())
    {
        checkCompletedTests();
        usleep(100000);  // 100ms
    }
}

void GPUScheduler::submitJob(const TestJob& job)
{
    std::lock_guard<std::mutex> lock(statsMutex_);

    TestJob jobCopy = job;
    jobCopy.jobId = nextJobId_++;
    jobCopy.submittedTime = std::chrono::steady_clock::now();

    pendingJobs_.push(jobCopy);
    totalSubmitted_++;

    if (config_.verboseLogging)
    {
        INFO("Job submitted: %s (ID: %d, GPUs: %d, Priority: %d)\n",
             jobCopy.testName.c_str(), jobCopy.jobId, jobCopy.numGPUsRequired, jobCopy.priority);
    }
}

void GPUScheduler::submitJobs(const std::vector<TestJob>& jobs)
{
    for (const auto& job : jobs)
    {
        submitJob(job);
    }
}

void GPUScheduler::preallocateUniqueIds(int poolSize)
{
    if (config_.verboseLogging)
    {
        INFO("Pre-generating pool of %d NCCL unique IDs to avoid serialization...\n", poolSize);
    }

    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < poolSize; ++i)
    {
        ncclUniqueId id;
        ncclResult_t result = ncclGetUniqueId(&id);
        if (result != ncclSuccess)
        {
            ERROR("Failed to generate NCCL unique ID #%d: %d\n", i, result);
            continue;
        }

        // Store as raw bytes
        std::vector<char> idBytes(NCCL_UNIQUE_ID_BYTES);
        memcpy(idBytes.data(), &id, NCCL_UNIQUE_ID_BYTES);
        uniqueIdPool_.push_back(idBytes);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (config_.verboseLogging)
    {
        INFO("Generated %zu unique IDs in %ld ms (avg %.1f ms per ID)\n",
             uniqueIdPool_.size(), duration.count(),
             uniqueIdPool_.empty() ? 0.0 : (double)duration.count() / uniqueIdPool_.size());
    }
}

std::string GPUScheduler::getNextUniqueId()
{
    std::lock_guard<std::mutex> lock(poolMutex_);

    if (nextUniqueIdIndex_ >= uniqueIdPool_.size())
    {
        ERROR("Unique ID pool exhausted! Allocated %zu but need more.\n", uniqueIdPool_.size());
        return "";
    }

    const std::vector<char>& idBytes = uniqueIdPool_[nextUniqueIdIndex_++];

    // Convert to hex string for environment variable
    std::string hexStr;
    hexStr.reserve(NCCL_UNIQUE_ID_BYTES * 2);
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i)
    {
        char hex[3];
        sprintf(hex, "%02x", (unsigned char)idBytes[i]);
        hexStr += hex;
    }

    return hexStr;
}


void GPUScheduler::startAsyncIdGeneration(int totalNeeded)
{
    if (config_.verboseLogging)
    {
        INFO("Generating pool of %d unique IDs in separate process...\n", totalNeeded);
    }

    auto startTime = std::chrono::steady_clock::now();

    // Fork a separate process to generate IDs
    // This prevents GPU context initialization in the parent process
    int pipefd[2];
    if (pipe(pipefd) == -1)
    {
        ERROR("Failed to create pipe for unique ID generation\n");
        return;
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        // Child process: Generate all unique IDs
        close(pipefd[0]);  // Close read end

        for (int i = 0; i < totalNeeded; ++i)
        {
            ncclUniqueId id;
            ncclResult_t result = ncclGetUniqueId(&id);
            if (result != ncclSuccess)
            {
                ERROR("ID generator process: Failed to generate ID #%d: %d\n", i, result);
                _exit(1);
            }

            // Write ID to pipe
            ssize_t written = write(pipefd[1], &id, sizeof(ncclUniqueId));
            if (written != sizeof(ncclUniqueId))
            {
                ERROR("ID generator process: Failed to write ID #%d to pipe\n", i);
                _exit(1);
            }
        }

        close(pipefd[1]);
        _exit(0);  // Success
    }
    else if (pid > 0)
    {
        // Parent process: Read IDs from pipe
        close(pipefd[1]);  // Close write end

        for (int i = 0; i < totalNeeded; ++i)
        {
            ncclUniqueId id;
            ssize_t bytesRead = read(pipefd[0], &id, sizeof(ncclUniqueId));
            if (bytesRead != sizeof(ncclUniqueId))
            {
                ERROR("Failed to read unique ID #%d from pipe\n", i);
                break;
            }

            std::vector<char> idBytes(NCCL_UNIQUE_ID_BYTES);
            memcpy(idBytes.data(), &id, NCCL_UNIQUE_ID_BYTES);
            uniqueIdPool_.push_back(idBytes);
        }

        close(pipefd[0]);

        // Wait for child process to exit
        int status;
        waitpid(pid, &status, 0);

        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        if (config_.verboseLogging)
        {
            INFO("Generated %zu unique IDs in %ld ms (%.1f ms per ID, in separate process)\n",
                 uniqueIdPool_.size(), duration.count(),
                 uniqueIdPool_.empty() ? 0.0 : (double)duration.count() / uniqueIdPool_.size());
        }
    }
    else
    {
        ERROR("Failed to fork ID generator process\n");
        close(pipefd[0]);
        close(pipefd[1]);
    }
}

void GPUScheduler::stopAsyncIdGeneration()
{
    if (uniqueIdGeneratorThread_.joinable())
    {
        stopIdGeneration_.store(true);
        uniqueIdGeneratorThread_.join();
    }
}

bool GPUScheduler::canAllocateGPUs(int numGPUs) const
{
    if (numGPUs > config_.totalGPUs)
        return false;

    int availableGPUs = 0;
    for (bool inUse : gpuInUse_)
    {
        if (!inUse)
            availableGPUs++;
    }

    return availableGPUs >= numGPUs;
}

std::vector<int> GPUScheduler::allocateGPUs(int numGPUs)
{
    std::vector<int> allocated;
    allocated.reserve(numGPUs);

    for (int i = 0; i < config_.totalGPUs && allocated.size() < numGPUs; ++i)
    {
        if (!gpuInUse_[i])
        {
            gpuInUse_[i] = true;
            allocated.push_back(i);
        }
    }

    if (config_.verboseLogging && !allocated.empty())
    {
        std::stringstream ss;
        ss << "Allocated GPUs: ";
        for (size_t i = 0; i < allocated.size(); ++i)
        {
            if (i > 0) ss << ",";
            ss << allocated[i];
        }
        INFO("%s\n", ss.str().c_str());
    }

    return allocated;
}

void GPUScheduler::releaseGPUs(const std::vector<int>& gpus)
{
    for (int gpuId : gpus)
    {
        if (gpuId >= 0 && gpuId < config_.totalGPUs)
        {
            gpuInUse_[gpuId] = false;
        }
    }

    if (config_.verboseLogging && !gpus.empty())
    {
        std::stringstream ss;
        ss << "Released GPUs: ";
        for (size_t i = 0; i < gpus.size(); ++i)
        {
            if (i > 0) ss << ",";
            ss << gpus[i];
        }
        INFO("%s\n", ss.str().c_str());
    }
}

bool GPUScheduler::launchTest(const TestJob& job, const std::vector<int>& gpus)
{
    if (config_.verboseLogging)
    {
        INFO("Launching test: %s (Job ID: %d) on %zu GPUs\n",
             job.testName.c_str(), job.jobId, gpus.size());
    }

    // Flush output before fork
    fflush(NULL);

    pid_t pid = fork();

    // Get a pre-generated unique ID from the pool
    std::string uniqueIdHex = getNextUniqueId();
    if (uniqueIdHex.empty())
    {
        ERROR("Failed to get unique ID from pool for test %s\n", job.testName.c_str());
        return false;
    }

    if (pid == 0)
    {
        // Child process
        // NOTE: We do NOT set HIP_VISIBLE_DEVICES here!
        // Setting it causes issues because HIP/RCCL may initialize before the env var takes effect.
        // Instead, we pass the physical GPU IDs directly to the test via the assignedGPUs parameter.

        // Set pre-generated NCCL unique ID for this test
        setenv("UT_RCCL_UNIQUE_ID", uniqueIdHex.c_str(), 1);

        if (config_.verboseLogging)
        {
            std::stringstream gpuList;
            for (size_t i = 0; i < gpus.size(); ++i)
            {
                if (i > 0) gpuList << ",";
                gpuList << gpus[i];
            }
            INFO("Child process %d: Using physical GPUs: %s, uniqueId: %s\n",
                 getpid(), gpuList.str().c_str(), uniqueIdHex.substr(0, 16).c_str());
        }

        // NOTE: We do NOT call ncclGetUniqueId() here!
        // TestBed has its own mechanism where it forks all rank children first,
        // then asks child 0 to generate the unique ID, then broadcasts it.
        // If we call ncclGetUniqueId() here, we initialize the GPU/HIP context
        // BEFORE TestBed forks, causing all rank children to inherit the context
        // which leads to GPU memory conflicts and OOM errors.

        // Execute the test function with the GPU assignment
        int exitCode = 0;
        try
        {
            job.testFunction(gpus);
            // Check if GoogleTest detected any failures
            // Note: GoogleTest failures don't throw exceptions
            if (testing::Test::HasFailure())
            {
                exitCode = 1;
            }
        }
        catch (...)
        {
            exitCode = 1;
        }

        // Use exit() instead of _exit() to ensure C++ destructors run
        // This is critical for TestBed cleanup which releases GPU memory
        exit(exitCode);
    }
    else if (pid > 0)
    {
        // Parent process - record the running test
        RunningTest runningTest(job, gpus, pid);
        runningTests_.push_back(runningTest);
        return true;
    }
    else
    {
        // Fork failed
        ERROR("Failed to fork process for test: %s\n", job.testName.c_str());
        return false;
    }
}

int GPUScheduler::schedulePendingJobs()
{
    if (!config_.enableParallelExecution)
        return 0;

    int jobsLaunched = 0;
    std::lock_guard<std::mutex> lock(statsMutex_);

    // Try to launch jobs while we have capacity and pending jobs
    while (!pendingJobs_.empty() &&
           runningTests_.size() < static_cast<size_t>(config_.maxConcurrentTests))
    {
        TestJob nextJob = pendingJobs_.top();

        // Check if this test requires more GPUs than the system has
        if (nextJob.numGPUsRequired > config_.totalGPUs)
        {
            // Skip tests that require more GPUs than available on this machine
            pendingJobs_.pop();
            if (config_.verboseLogging)
            {
                INFO("Skipping test %s: requires %d GPUs but system has %d\n",
                     nextJob.testName.c_str(), nextJob.numGPUsRequired, config_.totalGPUs);
            }
            continue;
        }

        // Check if we can allocate the required GPUs right now
        if (!canAllocateGPUs(nextJob.numGPUsRequired))
        {
            // Can't allocate resources for the highest priority job right now
            // Wait for some GPUs to be released
            break;
        }

        // Remove from pending queue
        pendingJobs_.pop();

        // Allocate GPUs
        std::vector<int> gpus = allocateGPUs(nextJob.numGPUsRequired);

        if (gpus.size() != static_cast<size_t>(nextJob.numGPUsRequired))
        {
            // Allocation failed unexpectedly
            ERROR("GPU allocation failed for job %s\n", nextJob.testName.c_str());
            releaseGPUs(gpus);
            continue;
        }

        // Launch the test
        if (launchTest(nextJob, gpus))
        {
            jobsLaunched++;
        }
        else
        {
            // Launch failed, release GPUs
            releaseGPUs(gpus);
        }
    }

    return jobsLaunched;
}

int GPUScheduler::checkCompletedTests()
{
    int completedCount = 0;
    auto now = std::chrono::steady_clock::now();

    // Check each running test
    for (auto it = runningTests_.begin(); it != runningTests_.end(); )
    {
        int status;
        pid_t result = waitpid(it->processId, &status, WNOHANG);

        if (result > 0)
        {
            // Process has completed
            bool success = (WIFEXITED(status) && WEXITSTATUS(status) == 0);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->startTime);

            if (config_.verboseLogging)
            {
                INFO("Test completed: %s (Job ID: %d, Duration: %ld ms, Status: %s)\n",
                     it->job.testName.c_str(), it->job.jobId, duration.count(),
                     success ? "PASSED" : "FAILED");
            }

            // Update GPU busy time
            for (int gpuId : it->assignedGPUs)
            {
                gpuBusyTime_[gpuId] += duration;
            }

            // Release GPUs
            releaseGPUs(it->assignedGPUs);

            // Record completion
            recordCompletion(*it, success);

            // Remove from running tests
            it = runningTests_.erase(it);
            completedCount++;
        }
        else if (result == 0)
        {
            // Process still running
            ++it;
        }
        else
        {
            // Error in waitpid
            ERROR("waitpid error for process %d: %s\n", it->processId, strerror(errno));
            releaseGPUs(it->assignedGPUs);
            it = runningTests_.erase(it);
        }
    }

    return completedCount;
}

void GPUScheduler::recordCompletion(const RunningTest& test, bool success)
{
    std::lock_guard<std::mutex> lock(statsMutex_);

    if (success)
        totalCompleted_++;
    else
        totalFailed_++;
}

void GPUScheduler::runUntilComplete()
{
    if (config_.verboseLogging)
    {
        INFO("Starting GPU scheduler execution loop\n");
    }

    while (!allJobsComplete())
    {
        // Check for completed tests
        checkCompletedTests();

        // Try to schedule pending jobs
        schedulePendingJobs();

        // Small sleep to avoid busy waiting
        usleep(10000);  // 10ms
    }

    if (config_.verboseLogging)
    {
        INFO("All jobs completed\n");
        printState();
    }
}

bool GPUScheduler::allJobsComplete() const
{
    std::lock_guard<std::mutex> lock(statsMutex_);
    return pendingJobs_.empty() && runningTests_.empty();
}

double GPUScheduler::calculateGPUUtilization() const
{
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - schedulerStartTime_);

    if (totalTime.count() == 0)
        return 0.0;

    std::chrono::milliseconds totalBusyTime(0);
    for (const auto& busyTime : gpuBusyTime_)
    {
        totalBusyTime += busyTime;
    }

    // Average utilization across all GPUs
    double utilization = (100.0 * totalBusyTime.count()) /
                        (totalTime.count() * config_.totalGPUs);

    return utilization;
}

GPUScheduler::Statistics GPUScheduler::getStatistics() const
{
    std::lock_guard<std::mutex> lock(statsMutex_);

    Statistics stats;
    stats.totalJobsSubmitted = totalSubmitted_;
    stats.totalJobsCompleted = totalCompleted_;
    stats.totalJobsFailed = totalFailed_;
    stats.currentlyRunning = runningTests_.size();
    stats.currentlyPending = pendingJobs_.size();
    stats.averageGPUUtilization = calculateGPUUtilization();
    stats.totalExecutionTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - schedulerStartTime_);

    return stats;
}

void GPUScheduler::printState() const
{
    std::lock_guard<std::mutex> lock(statsMutex_);

    INFO("=== GPU Scheduler State ===\n");
    INFO("Submitted: %d, Completed: %d, Failed: %d\n",
         totalSubmitted_, totalCompleted_, totalFailed_);
    INFO("Running: %zu, Pending: %zu\n",
         runningTests_.size(), pendingJobs_.size());

    // GPU utilization
    std::stringstream gpuState;
    gpuState << "GPU Status: [";
    for (size_t i = 0; i < gpuInUse_.size(); ++i)
    {
        gpuState << (gpuInUse_[i] ? "X" : "-");
    }
    gpuState << "]";
    INFO("%s\n", gpuState.str().c_str());

    INFO("Average GPU Utilization: %.1f%%\n", calculateGPUUtilization());
    INFO("===========================\n");
}

} // namespace RcclUnitTesting
