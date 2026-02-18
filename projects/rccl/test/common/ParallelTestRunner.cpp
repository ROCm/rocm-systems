/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ParallelTestRunner.hpp"
#include "ErrCode.hpp"
#include <cstdlib>
#include <cmath>

namespace RcclUnitTesting
{

ParallelTestRunner::ParallelTestRunner()
    : envVars_()
{
    initializeConfigFromEnvVars();
}

ParallelTestRunner::ParallelTestRunner(const EnvVars& envVars)
    : envVars_(envVars)
{
    initializeConfigFromEnvVars();
}

void ParallelTestRunner::initializeConfigFromEnvVars()
{
    // Read configuration from environment variables
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    config_.enableParallelExecution = (parallelEnv != nullptr && std::atoi(parallelEnv) != 0);

    const char* maxTestsEnv = std::getenv("UT_MAX_PARALLEL_TESTS");
    config_.maxConcurrentTests = maxTestsEnv ? std::atoi(maxTestsEnv) : 8;

    // Use the detected GPU count from EnvVars
    config_.totalGPUs = envVars_.maxGpus;

    const char* verboseEnv = std::getenv("UT_PARALLEL_VERBOSE");
    config_.verboseLogging = (verboseEnv != nullptr && std::atoi(verboseEnv) != 0) || envVars_.verbose;

    // GPU sharing is generally not safe for RCCL tests
    config_.allowGPUSharing = false;

    if (config_.verboseLogging)
    {
        INFO("ParallelTestRunner configuration:\n");
        INFO("  Parallel execution: %s\n", config_.enableParallelExecution ? "enabled" : "disabled");
        INFO("  Max concurrent tests: %d\n", config_.maxConcurrentTests);
        INFO("  Total GPUs: %d\n", config_.totalGPUs);
    }
}

int ParallelTestRunner::calculateDefaultPriority(int numGPUs) const
{
    // Priority strategy: Schedule larger GPU tests first (bin-packing optimization)
    // This helps avoid fragmentation where we can't fit large tests later
    // Priority range: 100 (1 GPU) down to 0 (8+ GPUs)
    return 100 - (numGPUs * 10);
}

void ParallelTestRunner::registerTest(const std::string& testName,
                                     int numGPUs,
                                     std::function<void()> testFunc,
                                     int priority)
{
    // Wrap the simple test function to ignore GPU assignments
    auto wrappedFunc = [testFunc](const std::vector<int>& /*gpus*/) {
        testFunc();
    };

    registerTestWithGPUInfo(testName, numGPUs, wrappedFunc, priority);
}

void ParallelTestRunner::registerTestWithGPUInfo(const std::string& testName,
                                                int numGPUs,
                                                std::function<void(const std::vector<int>&)> testFunc,
                                                int priority)
{
    // Use default priority calculation if not specified
    if (priority < 0)
    {
        priority = calculateDefaultPriority(numGPUs);
    }

    TestJob job(testName, numGPUs, testFunc, priority);
    registeredTests_.push_back(job);

    if (config_.verboseLogging)
    {
        INFO("Registered test: %s (GPUs: %d, Priority: %d)\n",
             testName.c_str(), numGPUs, priority);
    }
}

bool ParallelTestRunner::executeAll()
{
    if (registeredTests_.empty())
    {
        INFO("No tests registered\n");
        return true;
    }

    INFO("Executing %zu tests with GPU-aware scheduling\n", registeredTests_.size());

    // Check if parallel execution is disabled
    if (!config_.enableParallelExecution)
    {
        INFO("Parallel execution disabled - running tests sequentially\n");

        bool allPassed = true;
        for (const auto& test : registeredTests_)
        {
            INFO("Running test: %s (%d GPUs)\n", test.testName.c_str(), test.numGPUsRequired);

            // For sequential execution, just use GPUs 0..N-1
            std::vector<int> gpus(test.numGPUsRequired);
            for (int i = 0; i < test.numGPUsRequired; ++i)
            {
                gpus[i] = i;
            }

            try
            {
                test.testFunction(gpus);
                INFO("Test PASSED: %s\n", test.testName.c_str());
            }
            catch (const std::exception& e)
            {
                ERROR("Test FAILED: %s - %s\n", test.testName.c_str(), e.what());
                allPassed = false;
            }
            catch (...)
            {
                ERROR("Test FAILED: %s - unknown exception\n", test.testName.c_str());
                allPassed = false;
            }
        }

        return allPassed;
    }

    // Use GPU scheduler for parallel execution
    GPUScheduler scheduler(config_);

    // Submit all jobs to the scheduler
    scheduler.submitJobs(registeredTests_);

    // Run until all complete
    auto startTime = std::chrono::steady_clock::now();
    scheduler.runUntilComplete();
    auto endTime = std::chrono::steady_clock::now();

    // Get statistics
    auto stats = scheduler.getStatistics();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    INFO("\n");
    INFO("=============== GPU Scheduling Statistics ===============\n");
    INFO("Total tests executed: %d\n", stats.totalJobsSubmitted);
    INFO("Tests passed: %d\n", stats.totalJobsCompleted);
    INFO("Tests failed: %d\n", stats.totalJobsFailed);
    INFO("Total execution time: %ld ms\n", totalTime.count());
    INFO("Average GPU utilization: %.1f%%\n", stats.averageGPUUtilization);
    INFO("========================================================\n");

    return stats.totalJobsFailed == 0;
}

void ParallelTestRunner::clear()
{
    registeredTests_.clear();
}

size_t ParallelTestRunner::getTestCount() const
{
    return registeredTests_.size();
}

void ParallelTestRunner::printStatistics() const
{
    INFO("Registered tests: %zu\n", registeredTests_.size());

    // Group by GPU count
    std::map<int, int> testsByGPUCount;
    for (const auto& test : registeredTests_)
    {
        testsByGPUCount[test.numGPUsRequired]++;
    }

    INFO("Tests by GPU count:\n");
    for (const auto& entry : testsByGPUCount)
    {
        INFO("  %d GPUs: %d tests\n", entry.first, entry.second);
    }
}

void ParallelTestRunner::setParallelExecution(bool enabled)
{
    config_.enableParallelExecution = enabled;
}

void ParallelTestRunner::setMaxConcurrentTests(int maxTests)
{
    config_.maxConcurrentTests = maxTests;
}

void ParallelTestRunner::setVerboseLogging(bool enabled)
{
    config_.verboseLogging = enabled;
}

// TestSweepBuilder implementation

TestSweepBuilder::TestSweepBuilder(ParallelTestRunner& runner)
    : runner_(runner)
{
}

void TestSweepBuilder::registerSweep(const std::string& baseTestName,
                                    int minGPUs,
                                    int maxGPUs,
                                    std::function<std::function<void()>(int)> testFactory)
{
    for (int numGPUs = minGPUs; numGPUs <= maxGPUs; ++numGPUs)
    {
        std::string testName = baseTestName + "_" + std::to_string(numGPUs) + "GPU";
        auto testFunc = testFactory(numGPUs);
        runner_.registerTest(testName, numGPUs, testFunc);
    }
}

void TestSweepBuilder::registerPow2Sweep(const std::string& baseTestName,
                                        int minGPUs,
                                        int maxGPUs,
                                        std::function<std::function<void()>(int)> testFactory)
{
    // Start with the smallest power of 2 >= minGPUs
    int numGPUs = 1;
    while (numGPUs < minGPUs)
        numGPUs *= 2;

    // Register tests for each power of 2 up to maxGPUs
    while (numGPUs <= maxGPUs)
    {
        std::string testName = baseTestName + "_" + std::to_string(numGPUs) + "GPU";
        auto testFunc = testFactory(numGPUs);
        runner_.registerTest(testName, numGPUs, testFunc);
        numGPUs *= 2;
    }
}

} // namespace RcclUnitTesting
