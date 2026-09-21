/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include <hip/hip_runtime.h>
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

namespace RcclUnitTesting
{
  // Stress coverage for the parallelized TestBed::DestroyComms teardown
  // (ROCM-25953): DestroyComms now broadcasts the destroy command to every child
  // process first and collects the acknowledgements in a second pass, so the
  // children tear their communicators down concurrently instead of one at a time.
  // These tests drive that path repeatedly, in multi-process mode (one child per
  // GPU) where the two-pass ordering actually matters, and in both blocking and
  // non-blocking modes (the child-side DestroyComms branches on useBlocking).
  namespace
  {
    // Returns true if ncclFloat32 is available under the current UT_DATATYPES.
    // Checked once in each test body so the test can GTEST_SKIP() explicitly
    // rather than silently passing as a no-op when the datatype is excluded.
    bool float32Supported(TestBed& testBed)
    {
      std::vector<ncclDataType_t> dataTypes;
      testBed.GetSupportedDataTypes(dataTypes, {ncclFloat32});
      return !dataTypes.empty();
    }

    // Run `iterations` init / collective / destroy cycles over `totalRanks`
    // ranks, one child process per rank. Each cycle ends in DestroyComms, the
    // path under test. A leaked pipe fd or child handle, or any mismatch between
    // the broadcast pass and the ack-collection pass, shows up as a hang or a
    // failure once enough cycles accumulate.
    void RunTeardownCycles(TestBed& testBed,
                           int  const totalRanks,
                           bool const useBlocking,
                           int  const iterations,
                           bool&      isCorrect)
    {
      size_t const numElements   = 32 * 1024;
      bool   const inPlace       = false;
      bool   const useManagedMem = false;
      int    const numProcesses  = totalRanks;  // one child process per rank
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      for (int iter = 0; iter < iterations && isCorrect; ++iter)
      {
        testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks,
                                                    gpuPriorityOrder),
                          1, 1, 1, useBlocking);

        OptionalColArgs options;
        options.redOp = ncclSum;
        testBed.SetCollectiveArgs(ncclCollAllReduce, ncclFloat32,
                                  numElements, numElements, options);
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
        testBed.DestroyComms();
      }
    }
  }

  // Repeated multi-process teardown in both blocking and non-blocking modes.
  TEST(Teardown, RepeatedDestroyComms)
  {
    TestBed testBed;
    if (testBed.ev.maxGpus < 2)
      GTEST_SKIP() << "Teardown stress requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    if (!(testBed.ev.processMask & (1 << 1)))
      GTEST_SKIP() << "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
    if (!float32Supported(testBed))
      GTEST_SKIP() << "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";

    bool isCorrect = true;
    for (bool useBlocking : {true, false})
      RunTeardownCycles(testBed, testBed.ev.maxGpus, useBlocking,
                        /*iterations*/ 3, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // Teardown across a varying number of child processes within one test, so
  // DestroyComms is exercised with different numActiveChildren values back to
  // back. Catches assumptions that only hold for a fixed child count.
  TEST(Teardown, DestroyCommsVaryingChildCount)
  {
    TestBed testBed;
    if (testBed.ev.maxGpus < 2)
      GTEST_SKIP() << "Teardown stress requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    if (!(testBed.ev.processMask & (1 << 1)))
      GTEST_SKIP() << "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
    if (!float32Supported(testBed))
      GTEST_SKIP() << "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";

    bool isCorrect = true;
    for (int ranks = testBed.ev.maxGpus; ranks >= 2 && isCorrect; ranks /= 2)
      RunTeardownCycles(testBed, ranks, /*useBlocking*/ true,
                        /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // A worker can exit before teardown after a HIP/RCCL failure. Its closed
  // command pipe must not prevent the parent from reaping it and resetting the
  // TestBed state.
  TEST(Teardown, AlreadyExitedChildCleanup)
  {
    TestBed testBed;
    testBed.poolMode = false;
    testBed.configUsedPool = false;

    TestBedChild* child = new TestBedChild(0, false, 0, false);
    ASSERT_EQ(child->InitPipes(), TEST_SUCCESS);
    child->pid = fork();
    ASSERT_GE(child->pid, 0);
    if (child->pid == 0)
    {
      close(child->parentWriteFd);
      close(child->parentReadFd);
      close(child->childWriteFd);
      close(child->childReadFd);
      _exit(0);
    }

    close(child->childWriteFd);
    close(child->childReadFd);
    child->childWriteFd = -1;
    child->childReadFd = -1;
    testBed.childList = {child};
    testBed.numActiveChildren = 1;
    testBed.numActiveRanks = 1;

    // Wait until the worker has exited, but leave it for TestBed to reap.
    siginfo_t childInfo{};
    ASSERT_EQ(waitid(P_PID, child->pid, &childInfo, WEXITED | WNOWAIT), 0);
    pid_t const childPid = child->pid;

    testBed.TeardownOwnedChildList();

    EXPECT_TRUE(testBed.childList.empty());
    EXPECT_EQ(testBed.numActiveChildren, 0);
    EXPECT_EQ(testBed.numActiveRanks, 0);
    errno = 0;
    EXPECT_EQ(waitpid(childPid, nullptr, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }

  // Pool workers must start from a fresh process image even when an earlier
  // test initialized HIP in the parent process.
  TEST(Teardown, PoolAfterParentHipInitialization)
  {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    void* parentAllocation = nullptr;
    ASSERT_EQ(hipMalloc(&parentAllocation, 1), hipSuccess);
    ASSERT_EQ(hipFree(parentAllocation), hipSuccess);

    TestBed testBed;
    if (!testBed.poolMode)
      GTEST_SKIP() << "Requires communicator pooling (UT_COMM_POOL=1)";
    if (testBed.ev.maxGpus < 2)
      GTEST_SKIP() << "Teardown stress requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    if (!(testBed.ev.processMask & (1 << 1)))
      GTEST_SKIP() << "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
    if (!float32Supported(testBed))
      GTEST_SKIP() << "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";

    bool isCorrect = true;
    RunTeardownCycles(testBed, testBed.ev.maxGpus, /*useBlocking*/ true,
                      /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }
}
