/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

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
}
