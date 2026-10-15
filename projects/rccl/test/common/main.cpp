/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"

// DEBUG ONLY (revert before merge): a worker that dies on a signal otherwise only
// shows up as "Child N pipe closed unexpectedly" in the parent.
namespace
{
  int crashChildId = -1;

  void WorkerCrashHandler(int sig)
  {
    char msg[160];
    int const len = snprintf(msg, sizeof(msg), "[CHILD FATAL] child %d pid %d caught signal %d (%s); backtrace:\n",
                             crashChildId, static_cast<int>(getpid()), sig, strsignal(sig));
    if (len > 0) (void)!write(STDERR_FILENO, msg, static_cast<size_t>(len));
    void* frames[64];
    int const numFrames = backtrace(frames, 64);
    backtrace_symbols_fd(frames, numFrames, STDERR_FILENO);
    signal(sig, SIG_DFL);
    raise(sig);
  }

  void InstallWorkerCrashHandler(int childId)
  {
    crashChildId = childId;
    // backtrace() loads libgcc on first use, which is not async-signal-safe.
    void* warmup[1];
    (void)backtrace(warmup, 1);

    static char altStack[64 * 1024];
    stack_t ss = {};
    ss.ss_sp = altStack;
    ss.ss_size = sizeof(altStack);
    sigaltstack(&ss, nullptr);

    struct sigaction sa = {};
    sa.sa_handler = WorkerCrashHandler;
    sa.sa_flags = SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL})
      sigaction(sig, &sa, nullptr);
  }
}

int main(int argc, char **argv)
{
// 1. Intercept Child Worker Mode immediately at startup
    using RcclUnitTesting::TestBedChild;
    if (argc > TestBedChild::CHILD_ARG_FLAG && std::string(argv[TestBedChild::CHILD_ARG_FLAG]) == TestBedChild::kChildFlag) {
        // A worker must never fall through to RUN_ALL_TESTS: the parent would wait on it forever.
        if (argc != TestBedChild::NUM_CHILD_ARGS) {
            std::cerr << "[CHILD FATAL] Expected " << TestBedChild::NUM_CHILD_ARGS
                      << " worker arguments, got " << argc << std::endl;
            return 1;
        }
        int childId           = std::stoi(argv[TestBedChild::CHILD_ARG_ID]);
        int childReadFd       = std::stoi(argv[TestBedChild::CHILD_ARG_READ_FD]);
        int childWriteFd      = std::stoi(argv[TestBedChild::CHILD_ARG_WRITE_FD]);
        bool verbose          = (std::stoi(argv[TestBedChild::CHILD_ARG_VERBOSE]) != 0);
        int printValues       = std::stoi(argv[TestBedChild::CHILD_ARG_PRINT_VALUES]);
        bool useRankThreading = (std::stoi(argv[TestBedChild::CHILD_ARG_RANK_THREADING]) != 0);
        RcclUnitTesting::MemAllocType memAllocType = static_cast<RcclUnitTesting::MemAllocType>(
            std::stoi(argv[TestBedChild::CHILD_ARG_MEM_ALLOC_TYPE]));


        // Verify pipe file descriptors survived execl()
        if (fcntl(childReadFd, F_GETFD) == -1 || fcntl(childWriteFd, F_GETFD) == -1) {
            std::cerr << "[CHILD FATAL] Pipe FDs (" << childReadFd << ", " << childWriteFd
                      << ") are invalid or closed in child process " << childId << std::endl;
            return 1;
        }

        InstallWorkerCrashHandler(childId);

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
