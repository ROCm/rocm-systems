/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
#include "GlobalGPUScheduler.hpp"
#include <cstdlib>
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars ev;
  ev.ShowConfig();

  // Check if global scheduler is enabled (cross-suite parallelization)
  const char* globalSchedulerEnv = std::getenv("UT_GLOBAL_SCHEDULER");
  bool useGlobalScheduler = (globalSchedulerEnv != nullptr && std::atoi(globalSchedulerEnv) != 0);

  if (useGlobalScheduler)
  {
    // Initialize global GPU scheduler for cross-suite parallelization
    RcclUnitTesting::GPUSchedulingConfig config;
    config.totalGPUs = ev.maxGpus;
    config.enableParallelExecution = true;

    const char* maxTestsEnv = std::getenv("UT_MAX_PARALLEL_TESTS");
    config.maxConcurrentTests = maxTestsEnv ? std::atoi(maxTestsEnv) : 8;

    const char* verboseEnv = std::getenv("UT_PARALLEL_VERBOSE");
    config.verboseLogging = (verboseEnv != nullptr && std::atoi(verboseEnv) != 0) || ev.verbose;

    RcclUnitTesting::GlobalGPUScheduler::initialize(config);

    if (config.verboseLogging)
    {
      printf("[ INFO     ] Global GPU Scheduler enabled - jobs from all test suites will run in parallel\n");
      printf("[ INFO     ] Total GPUs: %d, Max concurrent tests: %d\n",
             config.totalGPUs, config.maxConcurrentTests);
    }
  }

  // Run all tests
  int retCode = RUN_ALL_TESTS();

  // Wait for all jobs to complete and shutdown global scheduler if enabled
  if (useGlobalScheduler)
  {
    RcclUnitTesting::GlobalGPUScheduler::waitForAllJobs();

    auto stats = RcclUnitTesting::GlobalGPUScheduler::getStatistics();

    printf("\n");
    printf("[ INFO     ] ============= Global GPU Scheduler Statistics =============\n");
    printf("[ INFO     ] Total jobs submitted: %d\n", stats.totalJobsSubmitted);
    printf("[ INFO     ] Completed: %d, Failed: %d\n", stats.totalJobsCompleted, stats.totalJobsFailed);
    printf("[ INFO     ] Total execution time: %ld ms (%.2f minutes)\n",
           stats.totalExecutionTime.count(), stats.totalExecutionTime.count() / 60000.0);
    printf("[ INFO     ] Average GPU utilization: %.1f%%\n", stats.averageGPUUtilization);
    printf("[ INFO     ] ==========================================================\n");

    RcclUnitTesting::GlobalGPUScheduler::shutdown();
  }

  printf("[ INFO     ] Total executed cases: %d\n", RcclUnitTesting::TestBed::NumTestsRun());

  // Show timing information
  if (ev.showTiming)
  {
    size_t totalTimeMsec = 0;
    fflush(stdout);
    printf("[ TIMING   ] %-20s: %-20s: %10s ms (%s)\n", "TEST SUITE", "TEST NAME", "TIME", "STATUS");
    auto unitTest = ::testing::UnitTest::GetInstance();
    for (int i = 0; i < unitTest->total_test_suite_count(); i++)
    {
      auto suiteInfo = unitTest->GetTestSuite(i);
      if (!suiteInfo->should_run()) continue;

      for (int j = 0; j < suiteInfo->total_test_count(); j++)
      {
        auto testInfo = suiteInfo->GetTestInfo(j);
        if (!testInfo->should_run()) continue;
        auto testResult = testInfo->result();
        if (testResult->Skipped()) continue;
        printf("[ TIMING   ] %-20s: %-20s: %10.2f sec (%4s)\n", testInfo->test_suite_name(), testInfo->name(), testResult->elapsed_time() / 1000.0, testResult->Passed() ? "PASS" : "FAIL");
      }
      printf("[ TIMING   ] %-20s: %-20s: %10.2f sec (%4s)\n", suiteInfo->name(), "TOTAL", suiteInfo->elapsed_time() / 1000.0, suiteInfo->Passed() ? "PASS" : "FAIL");
      totalTimeMsec += suiteInfo->elapsed_time();
    }
    printf("[ TIMING   ] Total time: %10.2f minutes\n", totalTimeMsec / (60 * 1000.0));
  }
  return retCode;
}
