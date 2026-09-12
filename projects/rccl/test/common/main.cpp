/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <fcntl.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
int main(int argc, char **argv)
{
// 1. Intercept Child Worker Mode immediately at startup
    if (argc >= 9 && std::string(argv[1]) == "--child") {
        int childId           = std::stoi(argv[2]);
        int childReadFd       = std::stoi(argv[3]);
        int childWriteFd      = std::stoi(argv[4]);
        bool verbose          = (std::stoi(argv[5]) != 0);
        int printValues       = std::stoi(argv[6]);
        bool useRankThreading = (std::stoi(argv[7]) != 0);
        RcclUnitTesting::MemAllocType memAllocType = static_cast<RcclUnitTesting::MemAllocType>(std::stoi(argv[8]));
        

        // Verify pipe file descriptors survived execl()
        if (fcntl(childReadFd, F_GETFD) == -1 || fcntl(childWriteFd, F_GETFD) == -1) {
            std::cerr << "[CHILD FATAL] Pipe FDs (" << childReadFd << ", " << childWriteFd 
                      << ") are invalid or closed in child process " << childId << std::endl;
            return 1;
        }

        // Instantiate child worker
        RcclUnitTesting::TestBedChild child(childId, verbose, printValues, useRankThreading);
        child.childReadFd   = childReadFd;
        child.childWriteFd  = childWriteFd;
        child.parentReadFd  = -1;
        child.parentWriteFd = -1;
        child.memAllocType  = memAllocType;

        // Enter worker execution loop
        child.StartExecutionLoop();
        return 0; // Exit cleanly when CHILD_STOP is received
    }

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
