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

namespace RcclUnitTesting
{

GPUScheduler::GPUScheduler(const GPUSchedulingConfig& config)
    : config_(config)
    , totalSubmitted_(0)
    , totalCompleted_(0)
    , totalFailed_(0)
    , nextJobId_(0)
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

    if (pid == 0)
    {
        // Child process
        // Set environment variables to restrict GPU visibility
        std::stringstream visibleDevices;
        for (size_t i = 0; i < gpus.size(); ++i)
        {
            if (i > 0) visibleDevices << ",";
            visibleDevices << gpus[i];
        }

        // Set both CUDA and HIP environment variables for compatibility
        setenv("CUDA_VISIBLE_DEVICES", visibleDevices.str().c_str(), 1);
        setenv("HIP_VISIBLE_DEVICES", visibleDevices.str().c_str(), 1);
        setenv("ROCR_VISIBLE_DEVICES", visibleDevices.str().c_str(), 1);

        if (config_.verboseLogging)
        {
            INFO("Child process %d: HIP_VISIBLE_DEVICES=%s\n", getpid(), visibleDevices.str().c_str());
        }

        // Execute the test function with the GPU assignment
        try
        {
            job.testFunction(gpus);
            _exit(0);  // Success
        }
        catch (...)
        {
            _exit(1);  // Failure
        }
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

        // Check if we can allocate the required GPUs
        if (!canAllocateGPUs(nextJob.numGPUsRequired))
        {
            // Can't allocate resources for the highest priority job
            // Don't try others as they might have even higher requirements
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
