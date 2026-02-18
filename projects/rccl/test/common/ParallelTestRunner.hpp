/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "GPUScheduler.hpp"
#include "EnvVars.hpp"
#include <functional>
#include <vector>
#include <string>

namespace RcclUnitTesting
{

/**
 * ParallelTestRunner integrates GPU-aware scheduling with the existing TestBed infrastructure.
 *
 * Usage:
 *   ParallelTestRunner runner;
 *
 *   // Register tests for different GPU counts
 *   runner.registerTest("AllReduce_1GPU", 1, []() { /* test code */ });
 *   runner.registerTest("AllReduce_2GPU", 2, []() { /* test code */ });
 *   runner.registerTest("AllReduce_8GPU", 8, []() { /* test code */ });
 *
 *   // Execute all tests with optimal GPU scheduling
 *   runner.executeAll();
 */
class ParallelTestRunner
{
public:
    ParallelTestRunner();
    explicit ParallelTestRunner(const EnvVars& envVars);

    /**
     * Register a test to be executed
     * @param testName Descriptive name for the test
     * @param numGPUs Number of GPUs required by this test
     * @param testFunc Function to execute the test
     * @param priority Higher values = higher priority (default: auto-calculated based on GPU count)
     */
    void registerTest(const std::string& testName,
                     int numGPUs,
                     std::function<void()> testFunc,
                     int priority = -1);

    /**
     * Register a test that needs to know which physical GPUs it's assigned
     * @param testName Descriptive name for the test
     * @param numGPUs Number of GPUs required by this test
     * @param testFunc Function to execute the test (receives assigned GPU IDs)
     * @param priority Higher values = higher priority (default: auto-calculated based on GPU count)
     */
    void registerTestWithGPUInfo(const std::string& testName,
                                int numGPUs,
                                std::function<void(const std::vector<int>&)> testFunc,
                                int priority = -1);

    /**
     * Execute all registered tests using GPU-aware scheduling
     * @return true if all tests passed, false otherwise
     */
    bool executeAll();

    /**
     * Clear all registered tests
     */
    void clear();

    /**
     * Get number of registered tests
     */
    size_t getTestCount() const;

    /**
     * Print scheduling statistics
     */
    void printStatistics() const;

    /**
     * Enable/disable parallel execution (for debugging)
     */
    void setParallelExecution(bool enabled);

    /**
     * Set maximum number of concurrent tests
     */
    void setMaxConcurrentTests(int maxTests);

    /**
     * Enable/disable verbose logging
     */
    void setVerboseLogging(bool enabled);

private:
    GPUSchedulingConfig config_;
    std::vector<TestJob> registeredTests_;
    EnvVars envVars_;

    int calculateDefaultPriority(int numGPUs) const;
    void initializeConfigFromEnvVars();
};

/**
 * Helper class to automatically create test jobs for a GPU count sweep.
 * This replaces the manual loop in RunSimpleSweep.
 */
class TestSweepBuilder
{
public:
    TestSweepBuilder(ParallelTestRunner& runner);

    /**
     * Register a test to run across multiple GPU counts
     * @param baseTestName Base name for the test (will be suffixed with GPU count)
     * @param minGPUs Minimum number of GPUs
     * @param maxGPUs Maximum number of GPUs
     * @param testFactory Function that creates a test for a given GPU count
     */
    void registerSweep(const std::string& baseTestName,
                      int minGPUs,
                      int maxGPUs,
                      std::function<std::function<void()>(int)> testFactory);

    /**
     * Register sweep with power-of-2 GPU counts only
     */
    void registerPow2Sweep(const std::string& baseTestName,
                          int minGPUs,
                          int maxGPUs,
                          std::function<std::function<void()>(int)> testFactory);

private:
    ParallelTestRunner& runner_;
};

} // namespace RcclUnitTesting
