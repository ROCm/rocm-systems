/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <mutex>
#include <string>
#include <sys/types.h>

namespace RcclUnitTesting
{

// Configuration for GPU scheduling behavior
struct GPUSchedulingConfig
{
    bool   enableParallelExecution;  // Enable parallel test execution
    int    maxConcurrentTests;       // Maximum number of tests running simultaneously
    int    totalGPUs;                // Total number of GPUs available
    bool   allowGPUSharing;          // Allow multiple tests to share GPUs (usually false for RCCL)
    bool   verboseLogging;           // Print scheduling decisions

    GPUSchedulingConfig()
        : enableParallelExecution(true)
        , maxConcurrentTests(8)
        , totalGPUs(8)
        , allowGPUSharing(false)
        , verboseLogging(false)
    {}
};

// Represents a test job to be scheduled
struct TestJob
{
    std::string              testName;
    int                      numGPUsRequired;
    std::function<void(const std::vector<int>&)> testFunction;  // Function that runs the test with assigned GPU IDs
    int                      priority;           // Higher priority runs first

    // Job metadata
    std::chrono::steady_clock::time_point submittedTime;
    int                      jobId;

    TestJob()
        : numGPUsRequired(1)
        , priority(0)
        , submittedTime(std::chrono::steady_clock::now())
        , jobId(-1)
    {}

    TestJob(const std::string& name, int gpus, std::function<void(const std::vector<int>&)> func, int prio = 0)
        : testName(name)
        , numGPUsRequired(gpus)
        , testFunction(func)
        , priority(prio)
        , submittedTime(std::chrono::steady_clock::now())
        , jobId(-1)
    {}
};

// Tracks a running test
struct RunningTest
{
    TestJob                  job;
    std::vector<int>         assignedGPUs;
    pid_t                    processId;
    std::chrono::steady_clock::time_point startTime;

    RunningTest()
        : processId(-1)
        , startTime(std::chrono::steady_clock::now())
    {}

    RunningTest(const TestJob& j, const std::vector<int>& gpus, pid_t pid)
        : job(j)
        , assignedGPUs(gpus)
        , processId(pid)
        , startTime(std::chrono::steady_clock::now())
    {}
};

// Manages GPU allocation and test scheduling
class GPUScheduler
{
public:
    GPUScheduler(const GPUSchedulingConfig& config = GPUSchedulingConfig());
    ~GPUScheduler();

    // Submit a test job to be scheduled
    void submitJob(const TestJob& job);

    // Submit multiple jobs at once
    void submitJobs(const std::vector<TestJob>& jobs);

    // Try to schedule and launch pending jobs
    // Returns number of jobs launched
    int schedulePendingJobs();

    // Check for completed tests and release their GPUs
    // Returns number of tests that completed
    int checkCompletedTests();

    // Run the scheduler loop until all jobs complete
    // This is the main entry point for automated execution
    void runUntilComplete();

    // Check if all jobs are complete
    bool allJobsComplete() const;

    // Get scheduling statistics
    struct Statistics
    {
        int totalJobsSubmitted;
        int totalJobsCompleted;
        int totalJobsFailed;
        int currentlyRunning;
        int currentlyPending;
        double averageGPUUtilization;
        std::chrono::milliseconds totalExecutionTime;
    };
    Statistics getStatistics() const;

    // Print current scheduler state
    void printState() const;

private:
    GPUSchedulingConfig config_;

    // Job queues (ordered by priority)
    struct JobComparator
    {
        bool operator()(const TestJob& a, const TestJob& b) const
        {
            // Higher priority first, then larger GPU requirements (bin-packing optimization)
            if (a.priority != b.priority)
                return a.priority < b.priority;  // Lower priority value = lower priority in queue
            return a.numGPUsRequired < b.numGPUsRequired;
        }
    };
    std::priority_queue<TestJob, std::vector<TestJob>, JobComparator> pendingJobs_;

    // Running tests
    std::vector<RunningTest> runningTests_;

    // GPU availability tracking
    std::vector<bool> gpuInUse_;  // gpuInUse_[i] = true if GPU i is currently allocated

    // Statistics
    mutable std::mutex statsMutex_;
    int totalSubmitted_;
    int totalCompleted_;
    int totalFailed_;
    std::chrono::steady_clock::time_point schedulerStartTime_;
    std::vector<std::chrono::milliseconds> gpuBusyTime_;  // Track how long each GPU has been busy

    // Job ID counter
    int nextJobId_;

    // Helper functions
    bool canAllocateGPUs(int numGPUs) const;
    std::vector<int> allocateGPUs(int numGPUs);
    void releaseGPUs(const std::vector<int>& gpus);
    bool launchTest(const TestJob& job, const std::vector<int>& gpus);
    void recordCompletion(const RunningTest& test, bool success);
    double calculateGPUUtilization() const;
};

} // namespace RcclUnitTesting
