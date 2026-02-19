/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "GlobalGPUScheduler.hpp"
#include "ErrCode.hpp"
#include <iostream>
#include <chrono>
#include <unistd.h>

namespace RcclUnitTesting
{

// Static member initialization
GPUScheduler* GlobalGPUScheduler::instance_ = nullptr;
std::mutex GlobalGPUScheduler::instanceMutex_;
std::thread GlobalGPUScheduler::schedulerThread_;
std::atomic<bool> GlobalGPUScheduler::running_{false};
std::mutex GlobalGPUScheduler::completionMutex_;
std::condition_variable GlobalGPUScheduler::completionCV_;

void GlobalGPUScheduler::initialize(const GPUSchedulingConfig& config)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (instance_ != nullptr)
    {
        WARN("GlobalGPUScheduler already initialized. Ignoring duplicate initialization.\n");
        return;
    }

    if (config.verboseLogging)
    {
        INFO("Initializing GlobalGPUScheduler with %d GPUs, max %d concurrent tests\n",
             config.totalGPUs, config.maxConcurrentTests);
    }

    // Create scheduler instance
    instance_ = new GPUScheduler(config);
    running_ = true;

    // Start background scheduler thread
    schedulerThread_ = std::thread(schedulerLoop);

    if (config.verboseLogging)
    {
        INFO("GlobalGPUScheduler initialized. Background scheduler thread started.\n");
    }
}

void GlobalGPUScheduler::shutdown()
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (instance_ == nullptr)
    {
        return;
    }

    auto config = instance_->getStatistics();
    bool verbose = false;  // Assume verbose if we have instance (check by stats existence)

    // Wait for all jobs to complete before shutting down
    while (!instance_->allJobsComplete())
    {
        usleep(100000);  // 100ms
    }

    // Stop the scheduler thread
    running_ = false;
    if (schedulerThread_.joinable())
    {
        schedulerThread_.join();
    }

    // Clean up
    delete instance_;
    instance_ = nullptr;
}

bool GlobalGPUScheduler::isInitialized()
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return instance_ != nullptr;
}

int GlobalGPUScheduler::submitJob(const TestJob& job)
{
    if (instance_ == nullptr)
    {
        ERROR("GlobalGPUScheduler not initialized. Call initialize() first.\n");
        return -1;
    }

    // Submit job to scheduler and get the assigned job ID
    int jobId = instance_->submitJob(job);

    return jobId;
}

std::vector<int> GlobalGPUScheduler::submitJobs(const std::vector<TestJob>& jobs)
{
    std::vector<int> jobIds;
    jobIds.reserve(jobs.size());

    for (const auto& job : jobs)
    {
        int jobId = submitJob(job);
        jobIds.push_back(jobId);
    }

    return jobIds;
}

void GlobalGPUScheduler::waitForJobs(const std::vector<int>& jobIds)
{
    if (instance_ == nullptr)
    {
        ERROR("GlobalGPUScheduler not initialized.\n");
        return;
    }

    if (jobIds.empty())
    {
        return;
    }

    // Wait until all specified jobs are complete
    std::unique_lock<std::mutex> lock(completionMutex_);
    completionCV_.wait(lock, [&jobIds]() {
        return areJobsComplete(jobIds);
    });
}

void GlobalGPUScheduler::waitForAllJobs()
{
    if (instance_ == nullptr)
    {
        return;
    }

    // Wait until scheduler is idle
    while (!instance_->allJobsComplete())
    {
        usleep(100000);  // 100ms
    }
}

GPUScheduler::Statistics GlobalGPUScheduler::getStatistics()
{
    if (instance_ == nullptr)
    {
        return GPUScheduler::Statistics{};
    }

    return instance_->getStatistics();
}

void GlobalGPUScheduler::printState()
{
    if (instance_ != nullptr)
    {
        instance_->printState();
    }
}

void GlobalGPUScheduler::schedulerLoop()
{
    // Continuously check for completed tests and schedule pending jobs
    while (running_ || !instance_->allJobsComplete())
    {
        // Check for completed tests and release their GPUs
        int completed = instance_->checkCompletedTests();

        // If jobs completed, notify waiters
        if (completed > 0)
        {
            completionCV_.notify_all();
        }

        // Try to schedule pending jobs
        instance_->schedulePendingJobs();

        // Small sleep to avoid busy-waiting
        usleep(10000);  // 10ms
    }
}

bool GlobalGPUScheduler::areJobsComplete(const std::vector<int>& jobIds)
{
    // Delegate to the scheduler's tracking
    return instance_->areJobsComplete(jobIds);
}

} // namespace RcclUnitTesting
