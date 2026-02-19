/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Parallel execution support for TestBed::RunSimpleSweep
 ************************************************************************/

#include "TestBed.hpp"
#include "GPUScheduler.hpp"
#include <functional>
#include <sstream>

namespace RcclUnitTesting
{

// Helper struct to capture all test parameters for a single GPU count iteration
struct SweepTestJob
{
    int numGpus;
    int isMultiProcess;
    int ranksPerGpu;
    std::vector<ncclFunc_t>     funcTypes;
    std::vector<ncclDataType_t> dataTypes;
    std::vector<ncclRedOp_t>    redOps;
    std::vector<int>            roots;
    std::vector<int>            numElements;
    std::vector<bool>           inPlaceList;
    std::vector<bool>           managedMemList;
    std::vector<bool>           useHipGraphList;
    bool                        enableSweep;
    // Only copy the fields we actually use from EnvVars
    bool                        showNames;
    bool                        verbose;
    int                         printValues;
    bool                        isGfx90;
};

// Execute a single test job (one GPU count configuration)
static void ExecuteSweepJob(const SweepTestJob& job, const std::vector<int>& physicalGpuIds)
{
    TestBed testBed;

    // Override GPU settings for this specific job
    testBed.ev.minGpus = job.numGpus;
    testBed.ev.maxGpus = job.numGpus;

    // Copy other env settings
    testBed.ev.showNames = job.showNames;
    testBed.ev.verbose = job.verbose;
    testBed.ev.printValues = job.printValues;

    bool isCorrect = true;

    // Single GPU count iteration (the inner loops from RunSimpleSweep)
    int const numChildren = job.isMultiProcess ? job.numGpus : 1;
    int const numRanks    = job.numGpus * job.ranksPerGpu;

    if (!job.enableSweep && (job.numGpus < 8 || numRanks < 8)) {
        return;  // Skip based on enableSweep logic
    }

    // Use physical GPU IDs directly (no HIP_VISIBLE_DEVICES remapping)
    const std::vector<int>& gpuPriorityOrder = physicalGpuIds.empty() ?
        testBed.ev.GetGpuPriorityOrder() : physicalGpuIds;

    if (testBed.ev.verbose) {
        INFO("Using physical GPU IDs: [");
        for (size_t i = 0; i < gpuPriorityOrder.size(); ++i) {
            if (i > 0) INFO(", ");
            INFO("%d", gpuPriorityOrder[i]);
        }
        INFO("]\n");
    }

    testBed.InitComms(testBed.GetDeviceIdsList(numChildren, job.numGpus, job.ranksPerGpu, gpuPriorityOrder));

    if (testing::Test::HasFailure())
    {
        return;
    }

    // Sort numElements in descending order
    std::vector<int> sortedN = job.numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());
    OptionalColArgs optionalArgs;

    for (int ftIdx = 0; ftIdx < job.funcTypes.size() && isCorrect; ++ftIdx)
    for (int dtIdx = 0; dtIdx < job.dataTypes.size() && isCorrect; ++dtIdx)
    {
        // Skip AllReduce FP8 test on 9 to 16 ranks (gfx90a)
        if (job.isGfx90 && numRanks > 8 && job.funcTypes[ftIdx] == ncclCollAllReduce
                      && (job.dataTypes[dtIdx] == ncclFloat8e4m3
                      || job.dataTypes[dtIdx] == ncclFloat8e5m2))
        {
            continue;
        }

        for (int rdIdx = 0; rdIdx < job.redOps.size() && isCorrect; ++rdIdx)
        for (int rtIdx = 0; rtIdx < job.roots.size() && isCorrect; ++rtIdx)
        for (int ipIdx = 0; ipIdx < job.inPlaceList.size() && isCorrect; ++ipIdx)
        for (int mmIdx = 0; mmIdx < job.managedMemList.size() && isCorrect; ++mmIdx)
        {
            for (int neIdx = 0; neIdx < job.numElements.size() && isCorrect; ++neIdx)
            {
                int numInputElements, numOutputElements;
                CollectiveArgs::GetNumElementsForFuncType(job.funcTypes[ftIdx],
                                                          sortedN[neIdx],
                                                          numRanks,
                                                          &numInputElements,
                                                          &numOutputElements);

                optionalArgs.redOp = job.redOps[rdIdx];
                optionalArgs.root = job.roots[rtIdx] % testBed.numActiveRanks;

                if (optionalArgs.useBias)
                {
                    optionalArgs.biasNumElements = numOutputElements;
                }

                testBed.SetCollectiveArgs(job.funcTypes[ftIdx],
                                        job.dataTypes[dtIdx],
                                        numInputElements,
                                        numOutputElements,
                                        optionalArgs);

                if (testing::Test::HasFailure())
                {
                    isCorrect = false;
                    continue;
                }

                // Only allocate once for largest size
                if (neIdx == 0)
                {
                    testBed.AllocateMem(job.inPlaceList[ipIdx], job.managedMemList[mmIdx]);
                    if (testing::Test::HasFailure())
                    {
                        isCorrect = false;
                        continue;
                    }
                }

                for (int hgIdx = 0; hgIdx < job.useHipGraphList.size() && isCorrect; ++hgIdx)
                {
                    bool canSkip = (neIdx != 0 && !job.inPlaceList[ipIdx] &&
                                    (job.funcTypes[ftIdx] == ncclCollBroadcast ||
                                     job.funcTypes[ftIdx] == ncclCollReduce ||
                                     job.funcTypes[ftIdx] == ncclCollAllReduce));

                    if (!canSkip) testBed.PrepareData();

                    if (testing::Test::HasFailure())
                    {
                        isCorrect = false;
                        continue;
                    }

                    std::string name = testBed.GetTestCaseName(job.numGpus, job.isMultiProcess,
                                                             job.funcTypes[ftIdx], job.dataTypes[dtIdx],
                                                             job.redOps[rdIdx], job.roots[rtIdx],
                                                             job.inPlaceList[ipIdx], job.managedMemList[mmIdx],
                                                             job.useHipGraphList[hgIdx], job.ranksPerGpu);

                    if (job.showNames)
                    {
                        INFO("%s [%9d elements]\n", name.c_str(), numInputElements);
                    }

                    std::vector<int> currentRanksEmpty = {};
                    testBed.ExecuteCollectives(currentRanksEmpty, -1, job.useHipGraphList[hgIdx]);

                    if (job.useHipGraphList[hgIdx])
                    {
                        testBed.LaunchGraphs();
                        testBed.DestroyGraphs();
                    }

                    if (testing::Test::HasFailure())
                    {
                        isCorrect = false;
                        continue;
                    }

                    testBed.ValidateResults(isCorrect);
                    if (!isCorrect)
                    {
                        ERROR("Incorrect output for %s\n", name.c_str());
                    }
                }
            }
            testBed.DeallocateMem();
        }
    }

    testBed.DestroyComms();

    // Return exit code based on test results
    // Note: Caller should use exit() not _exit() to ensure proper cleanup
    if (!isCorrect || testing::Test::HasFailure())
    {
        // Don't call exit here - let the caller handle it to ensure proper cleanup
        throw std::runtime_error("Test failed");
    }
}

void TestBed::RunSimpleSweepParallel(std::vector<ncclFunc_t>     const& funcTypes,
                                    std::vector<ncclDataType_t> const& tmpDataTypes,
                                    std::vector<ncclRedOp_t>    const& tmpRedOps,
                                    std::vector<int>            const& roots,
                                    std::vector<int>            const& numElements,
                                    std::vector<bool>           const& inPlaceList,
                                    std::vector<bool>           const& managedMemList,
                                    std::vector<bool>           const& useHipGraphList,
                                    bool                        const& enableSweep)
{
    // Check if parallel execution is enabled
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    bool useParallel = (parallelEnv != nullptr && std::atoi(parallelEnv) != 0);

    if (!useParallel)
    {
        // Fall back to original sequential implementation
        RunSimpleSweep(funcTypes, tmpDataTypes, tmpRedOps, roots, numElements,
                      inPlaceList, managedMemList, useHipGraphList, enableSweep);
        return;
    }

    // Filter supported data types and reduction ops
    std::vector<ncclDataType_t> dataTypes;
    this->GetSupportedDataTypes(dataTypes, tmpDataTypes);
    if (dataTypes.empty()) {
        GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    std::vector<ncclRedOp_t> redOps;
    this->GetSupportedRedOps(redOps, tmpRedOps);
    if (redOps.empty()) {
        GTEST_SKIP() << "Skipping... test reduction operations excluded by UT_REDOPS.";
    }

    // Create GPU scheduler
    GPUSchedulingConfig config;
    config.totalGPUs = ev.maxGpus;
    config.enableParallelExecution = true;

    const char* maxTestsEnv = std::getenv("UT_MAX_PARALLEL_TESTS");
    config.maxConcurrentTests = maxTestsEnv ? std::atoi(maxTestsEnv) : 8;

    const char* verboseEnv = std::getenv("UT_PARALLEL_VERBOSE");
    config.verboseLogging = (verboseEnv != nullptr && std::atoi(verboseEnv) != 0) || ev.verbose;

    GPUScheduler scheduler(config);

    if (config.verboseLogging)
    {
        INFO("Parallel GPU sweep enabled: running tests for %d GPU counts concurrently\n",
             ev.GetNumGpusList().size());
    }

    // Create test jobs for each GPU count (the outer loop from original RunSimpleSweep)
    for (int numGpus : ev.GetNumGpusList())
    for (int isMultiProcess : ev.GetIsMultiProcessList())
    for (int ranksPerGpu = 1; ranksPerGpu <= ev.maxRanksPerGpu; ++ranksPerGpu)
    {
        // Create a test job for this GPU configuration
        SweepTestJob job;
        job.numGpus = numGpus;
        job.isMultiProcess = isMultiProcess;
        job.ranksPerGpu = ranksPerGpu;
        job.funcTypes = funcTypes;
        job.dataTypes = dataTypes;
        job.redOps = redOps;
        job.roots = roots;
        job.numElements = numElements;
        job.inPlaceList = inPlaceList;
        job.managedMemList = managedMemList;
        job.useHipGraphList = useHipGraphList;
        job.enableSweep = enableSweep;
        // Copy only the specific fields we need (avoid copying entire EnvVars which triggers expensive constructor)
        job.showNames = ev.showNames;
        job.verbose = ev.verbose;
        job.printValues = ev.printValues;
        job.isGfx90 = ev.isGfx90;

        // Create descriptive test name
        std::stringstream testName;
        testName << "Sweep_" << numGpus << "GPU";
        if (isMultiProcess) testName << "_MP";
        else testName << "_SP";
        if (ranksPerGpu > 1) testName << "_" << ranksPerGpu << "RPG";

        // Submit as a TestJob to the scheduler
        TestJob testJob(
            testName.str(),
            numGpus,
            [job](const std::vector<int>& assignedGPUs) {
                // Pass physical GPU IDs directly to the test
                // No HIP_VISIBLE_DEVICES remapping - use actual physical IDs
                ExecuteSweepJob(job, assignedGPUs);
            },
            numGpus * 10  // Priority: higher GPU count = higher priority (greedy scheduling)
        );

        scheduler.submitJob(testJob);

        // Try to launch jobs immediately after submission (overlap submission and execution)
        scheduler.schedulePendingJobs();
    }

    // Run scheduler until all jobs complete (continues launching and monitoring)
    if (config.verboseLogging)
    {
        INFO("All jobs submitted. Waiting for remaining tests to complete.\n");
    }

    scheduler.runUntilComplete();

    // Get statistics
    auto stats = scheduler.getStatistics();

    if (config.verboseLogging)
    {
        INFO("\n");
        INFO("=============== Parallel Sweep Statistics ===============\n");
        INFO("Test configurations executed: %d\n", stats.totalJobsSubmitted);
        INFO("Passed: %d, Failed: %d\n", stats.totalJobsCompleted, stats.totalJobsFailed);
        INFO("Total execution time: %ld ms\n", stats.totalExecutionTime.count());
        INFO("Average GPU utilization: %.1f%%\n", stats.averageGPUUtilization);
        INFO("========================================================\n");
    }

    // Fail the test if any sweep jobs failed
    ASSERT_EQ(stats.totalJobsFailed, 0) << "Some sweep configurations failed";
}

} // namespace RcclUnitTesting
