/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars ev;
  ev.ShowConfig();
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
