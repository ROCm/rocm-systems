/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * This file demonstrates how to use GPU-aware parallel test execution
 * to maximize GPU utilization and reduce test time.
 ************************************************************************/

#include "TestBed.hpp"
#include "ParallelTestRunner.hpp"

namespace RcclUnitTesting
{

// Example 1: Basic parallel sweep using ParallelTestRunner
TEST(ParallelExecution, BasicSweep)
{
    // Skip if parallel execution is not enabled
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (!parallelEnv || std::atoi(parallelEnv) == 0)
    {
        GTEST_SKIP() << "Parallel execution not enabled (set UT_PARALLEL_TESTS=1)";
    }

    ParallelTestRunner runner;

    // Register tests for GPU counts 1 through 8
    for (int numGPUs = 1; numGPUs <= 8; ++numGPUs)
    {
        std::string testName = "AllReduce_" + std::to_string(numGPUs) + "GPU";

        runner.registerTest(testName, numGPUs, [numGPUs]() {
            TestBed testBed;

            // Configuration for AllReduce test
            std::vector<ncclFunc_t>     funcTypes       = {ncclCollAllReduce};
            std::vector<ncclDataType_t> dataTypes       = {ncclFloat32};
            std::vector<ncclRedOp_t>    redOps          = {ncclSum};
            std::vector<int>            roots           = {0};
            std::vector<int>            numElements     = {8192};  // Smaller for faster demo
            std::vector<bool>           inPlaceList     = {false};
            std::vector<bool>           managedMemList  = {false};
            std::vector<bool>           useHipGraphList = {false};

            // Force this specific GPU count
            testBed.ev.minGpus = numGPUs;
            testBed.ev.maxGpus = numGPUs;

            testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                  inPlaceList, managedMemList, useHipGraphList, false);
            testBed.Finalize();
        });
    }

    // Execute all tests with optimal GPU scheduling
    ASSERT_TRUE(runner.executeAll());
}

// Example 2: Using TestSweepBuilder for automatic sweep creation
TEST(ParallelExecution, AutomaticSweep)
{
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (!parallelEnv || std::atoi(parallelEnv) == 0)
    {
        GTEST_SKIP() << "Parallel execution not enabled (set UT_PARALLEL_TESTS=1)";
    }

    ParallelTestRunner runner;
    TestSweepBuilder sweepBuilder(runner);

    // Automatically create tests for GPU counts 1-8
    sweepBuilder.registerSweep("Broadcast_AutoSweep", 1, 8,
        [](int numGPUs) {
            return [numGPUs]() {
                TestBed testBed;

                std::vector<ncclFunc_t>     funcTypes       = {ncclCollBroadcast};
                std::vector<ncclDataType_t> dataTypes       = {ncclFloat32};
                std::vector<ncclRedOp_t>    redOps          = {ncclSum};
                std::vector<int>            roots           = {0};
                std::vector<int>            numElements     = {4096};
                std::vector<bool>           inPlaceList     = {false};
                std::vector<bool>           managedMemList  = {false};
                std::vector<bool>           useHipGraphList = {false};

                testBed.ev.minGpus = numGPUs;
                testBed.ev.maxGpus = numGPUs;

                testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                      inPlaceList, managedMemList, useHipGraphList, false);
                testBed.Finalize();
            };
        });

    ASSERT_TRUE(runner.executeAll());
}

// Example 3: Power-of-2 GPU sweep (1, 2, 4, 8)
TEST(ParallelExecution, PowerOf2Sweep)
{
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (!parallelEnv || std::atoi(parallelEnv) == 0)
    {
        GTEST_SKIP() << "Parallel execution not enabled (set UT_PARALLEL_TESTS=1)";
    }

    ParallelTestRunner runner;
    TestSweepBuilder sweepBuilder(runner);

    // Create tests only for power-of-2 GPU counts
    sweepBuilder.registerPow2Sweep("ReduceScatter_Pow2", 1, 8,
        [](int numGPUs) {
            return [numGPUs]() {
                TestBed testBed;

                std::vector<ncclFunc_t>     funcTypes       = {ncclCollReduceScatter};
                std::vector<ncclDataType_t> dataTypes       = {ncclFloat32};
                std::vector<ncclRedOp_t>    redOps          = {ncclSum};
                std::vector<int>            roots           = {0};
                std::vector<int>            numElements     = {4096};
                std::vector<bool>           inPlaceList     = {false};
                std::vector<bool>           managedMemList  = {false};
                std::vector<bool>           useHipGraphList = {false};

                testBed.ev.minGpus = numGPUs;
                testBed.ev.maxGpus = numGPUs;

                testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                      inPlaceList, managedMemList, useHipGraphList, false);
                testBed.Finalize();
            };
        });

    ASSERT_TRUE(runner.executeAll());
}

// Example 4: Mixed test types with custom priorities
TEST(ParallelExecution, CustomPriorities)
{
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (!parallelEnv || std::atoi(parallelEnv) == 0)
    {
        GTEST_SKIP() << "Parallel execution not enabled (set UT_PARALLEL_TESTS=1)";
    }

    ParallelTestRunner runner;

    // Register a critical 8-GPU test with highest priority
    runner.registerTest("CriticalTest_8GPU", 8, []() {
        TestBed testBed;
        // ... test implementation ...
        INFO("Running critical 8-GPU test\n");
    }, 1000);  // Very high priority

    // Register regular tests with default priorities
    for (int numGPUs : {1, 2, 4})
    {
        std::string name = "RegularTest_" + std::to_string(numGPUs) + "GPU";
        runner.registerTest(name, numGPUs, [numGPUs]() {
            INFO("Running regular %d-GPU test\n", numGPUs);
        });
    }

    ASSERT_TRUE(runner.executeAll());
}

// Example 5: GPU-aware test (knows which physical GPUs it's using)
TEST(ParallelExecution, GPUAwareTest)
{
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (!parallelEnv || std::atoi(parallelEnv) == 0)
    {
        GTEST_SKIP() << "Parallel execution not enabled (set UT_PARALLEL_TESTS=1)";
    }

    ParallelTestRunner runner;

    for (int numGPUs = 2; numGPUs <= 4; numGPUs += 2)
    {
        std::string name = "GPUAware_" + std::to_string(numGPUs) + "GPU";

        runner.registerTestWithGPUInfo(name, numGPUs,
            [numGPUs](const std::vector<int>& assignedGPUs) {
                INFO("Test %dGPU running on physical GPUs: ", numGPUs);
                for (int gpu : assignedGPUs)
                {
                    INFO("%d ", gpu);
                }
                INFO("\n");

                // Environment variables are automatically set:
                // HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES

                // Your test can use this information for verification
                // or specialized GPU topology testing
            });
    }

    ASSERT_TRUE(runner.executeAll());
}

// Example 6: Sequential fallback (when parallel execution is disabled)
TEST(ParallelExecution, SequentialFallback)
{
    ParallelTestRunner runner;

    // These tests will run sequentially if UT_PARALLEL_TESTS=0
    // or in parallel if UT_PARALLEL_TESTS=1

    for (int i = 1; i <= 4; ++i)
    {
        runner.registerTest("Test_" + std::to_string(i), 2, [i]() {
            INFO("Running test %d\n", i);
        });
    }

    // Execution mode is determined by environment variable
    ASSERT_TRUE(runner.executeAll());
}

} // namespace RcclUnitTesting
