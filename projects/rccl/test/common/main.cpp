/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// -------------------------------------------------------------------------------------
// PROOF OF CONCEPT: init-parallel / execute-serial pipeline.
//
// When UT_INIT_BARRIER=<path> is set, this process (a) warms up the one-time RCCL
// device-code-object load -- the dominant ~74% of comm-init -- by creating and destroying
// a throwaway 1-GPU communicator, then (b) writes "<path>.ready" and (c) blocks until a
// coordinator creates "<path>.go". A coordinator launches N such processes so their inits
// OVERLAP, then releases them to EXECUTE one at a time (serial), avoiding co-tenancy
// contention while still hiding the expensive load. No-op unless UT_INIT_BARRIER is set.
// (POC only -- productionize with an in-band signal instead of files.)
// -------------------------------------------------------------------------------------
static void ut_init_barrier_poc()
{
  const char* bar = getenv("UT_INIT_BARRIER");
  if (!bar) return;
  int dev = 0;
  const char* dv = getenv("UT_INIT_WARM_DEV");
  if (dv) dev = atoi(dv);
  (void)hipSetDevice(dev);
  ncclComm_t warm = nullptr;
  int devs[1] = { dev };
  if (ncclCommInitAll(&warm, 1, devs) == ncclSuccess && warm)
  {
    ncclCommDestroy(warm);
  }
  (void)hipDeviceSynchronize();
  std::string ready = std::string(bar) + ".ready";
  FILE* f = fopen(ready.c_str(), "w");
  if (f) { fputs("ready\n", f); fclose(f); }
  std::string go = std::string(bar) + ".go";
  while (access(go.c_str(), F_OK) != 0) { usleep(20000); }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars ev;
  ev.ShowConfig();
  ut_init_barrier_poc();   // POC: overlap init across processes, serialize execute
  int retCode = RUN_ALL_TESTS();
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
