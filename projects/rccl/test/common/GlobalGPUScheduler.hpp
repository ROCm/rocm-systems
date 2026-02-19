/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "GPUScheduler.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace RcclUnitTesting
{

/**
 * Global GPU Scheduler Singleton
 *
 * Provides a shared scheduler instance across all test suites, enabling
 * cross-suite parallelization. Jobs from different test suites (AllReduce,
 * AllGather, AllToAll, etc.) can run simultaneously on available GPUs.
 *
 * Features:
 * - Background scheduler thread that continuously launches jobs as they arrive
 * - Non-blocking job submission - tests submit and immediately return
 * - Per-test-suite waiting - each test waits only for its own jobs
 * - Automatic resource management - GPUs released as jobs complete
 *
 * Usage:
 *   // In main():
 *   GlobalGPUScheduler::initialize(config);
 *
 *   // In each test:
 *   auto jobIds = GlobalGPUScheduler::submitJobs(...);
 *   GlobalGPUScheduler::waitForJobs(jobIds);  // Wait for our jobs only
 *
 *   // In main() after tests:
 *   GlobalGPUScheduler::shutdown();
 */
class GlobalGPUScheduler
{
public:
    /**
     * Initialize the global scheduler with given configuration.
     * Must be called before any tests run (typically in main()).
     *
     * @param config GPU scheduling configuration
     */
    static void initialize(const GPUSchedulingConfig& config);

    /**
     * Shutdown the global scheduler and clean up resources.
     * Waits for all pending jobs to complete before shutting down.
     * Must be called after all tests finish (typically in main()).
     */
    static void shutdown();

    /**
     * Check if the global scheduler has been initialized.
     *
     * @return true if initialized, false otherwise
     */
    static bool isInitialized();

    /**
     * Submit a single job to the global scheduler.
     * Job execution starts immediately if resources are available.
     * Non-blocking - returns immediately with job ID.
     *
     * @param job Test job to submit
     * @return Job ID for tracking completion
     */
    static int submitJob(const TestJob& job);

    /**
     * Submit multiple jobs to the global scheduler.
     * Jobs execution starts immediately as resources become available.
     * Non-blocking - returns immediately with job IDs.
     *
     * @param jobs Vector of test jobs to submit
     * @return Vector of job IDs for tracking completion
     */
    static std::vector<int> submitJobs(const std::vector<TestJob>& jobs);

    /**
     * Wait for specific jobs to complete.
     * Blocks until all specified jobs finish (pass or fail).
     * Other jobs can continue running in parallel.
     *
     * @param jobIds Vector of job IDs to wait for
     */
    static void waitForJobs(const std::vector<int>& jobIds);

    /**
     * Wait for all submitted jobs to complete.
     * Blocks until the entire scheduler is idle.
     * Use this in main() before shutdown.
     */
    static void waitForAllJobs();

    /**
     * Get statistics from the global scheduler.
     *
     * @return Current scheduler statistics
     */
    static GPUScheduler::Statistics getStatistics();

    /**
     * Print current scheduler state (for debugging).
     */
    static void printState();

private:
    // Singleton instance
    static GPUScheduler* instance_;
    static std::mutex instanceMutex_;

    // Background scheduler thread
    static std::thread schedulerThread_;
    static std::atomic<bool> running_;

    // Job completion tracking
    static std::mutex completionMutex_;
    static std::condition_variable completionCV_;

    // Scheduler loop - runs in background thread
    static void schedulerLoop();

    // Check if specific jobs are complete
    static bool areJobsComplete(const std::vector<int>& jobIds);

    // Prevent instantiation
    GlobalGPUScheduler() = delete;
    ~GlobalGPUScheduler() = delete;
    GlobalGPUScheduler(const GlobalGPUScheduler&) = delete;
    GlobalGPUScheduler& operator=(const GlobalGPUScheduler&) = delete;
};

} // namespace RcclUnitTesting
