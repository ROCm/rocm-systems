/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

#include <chrono>
#include <thread>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{
  // Test various collectives using with non-blocking comms
  TEST(NonBlocking, SingleCalls)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t> const  funcTypes = {ncclCollBroadcast,
                                                ncclCollReduce,
                                                ncclCollAllGather,
                                                ncclCollReduceScatter,
                                                ncclCollAllReduce,
                                                ncclCollGather,
                                                ncclCollScatter};
    int        const  numElements   = 1048576;
    bool       const  inPlace       = false;
    bool       const  useManagedMem = false;
    bool       const  useBlocking   = false;

    OptionalColArgs options;
    options.redOp = ncclSum;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      // Initialize communicators in non-blocking mode
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), 1, 1, 1, useBlocking);

      // Loop over various collective functions
      for (auto funcType : funcTypes)
      {
        if (testBed.ev.showNames)
          TEST_INFO("%s %d-ranks Non-Blocking %s",
               isMultiProcess ? "MP" : "SP", totalRanks, ncclFuncNames[funcType]);

        int numInputElements;
        int numOutputElements;
        CollectiveArgs::GetNumElementsForFuncType(funcType,
                                                  numElements,
                                                  totalRanks,
                                                  &numInputElements,
                                                  &numOutputElements);

        testBed.SetCollectiveArgs(funcType,
                                  ncclFloat,
                                  numInputElements,
                                  numOutputElements,
                                  options);

        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  namespace
  {
    constexpr size_t kCount = 1024;

    // Polls a non-blocking communicator until its pending operation completes.
    ncclResult_t waitForComm(ncclComm_t comm)
    {
      ncclResult_t state = ncclInProgress;
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (state == ncclInProgress && std::chrono::steady_clock::now() < deadline) {
        if (ncclCommGetAsyncError(comm, &state) != ncclSuccess) return ncclInternalError;
        if (state == ncclInProgress) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      return state;
    }

    // One-rank non-blocking comm: optionally one rejected call, then an AllReduce, finalize and destroy.
    void singleRankNonBlocking(bool rejectOneCall)
    {
      HIP_CHECK(hipSetDevice(0));
      ncclUniqueId id;
      ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking     = 0;
      ncclComm_t comm     = nullptr;
      ncclResult_t res    = ncclCommInitRankConfig(&comm, 1, id, 0, &config);
      ASSERT_TRUE(res == ncclSuccess || res == ncclInProgress) << ncclGetErrorString(res);
      ASSERT_EQ(waitForComm(comm), ncclSuccess);

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));
      float* send = nullptr;
      float* recv = nullptr;
      HIP_CHECK(hipMalloc(&send, kCount * sizeof(float)));
      HIP_CHECK(hipMalloc(&recv, kCount * sizeof(float)));
      std::vector<float> in(kCount, 3.0f), out(kCount, 0.0f);
      HIP_CHECK(hipMemcpy(send, in.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(recv, 0, kCount * sizeof(float)));

      if (rejectOneCall) {
        // Peer 1 is out of range on a 1-rank comm, so ArgsCheck in ncclEnqueueCheck rejects the call.
        EXPECT_EQ(ncclSend(send, kCount, ncclFloat32, 1, comm, stream), ncclInvalidArgument);
        // NCCL #2459: the rejection must not become the comm's async state.
        ncclResult_t state = ncclInProgress;
        ASSERT_EQ(ncclCommGetAsyncError(comm, &state), ncclSuccess);
        ASSERT_EQ(state, ncclSuccess) << "ncclInvalidArgument was stored as the async error: "
                                      << ncclGetErrorString(state);
      }

      res = ncclAllReduce(send, recv, kCount, ncclFloat32, ncclSum, comm, stream);
      ASSERT_TRUE(res == ncclSuccess || res == ncclInProgress) << ncclGetErrorString(res);
      ASSERT_EQ(waitForComm(comm), ncclSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipMemcpy(out.data(), recv, kCount * sizeof(float), hipMemcpyDeviceToHost));
      EXPECT_EQ(in, out);

      res = ncclCommFinalize(comm);
      ASSERT_TRUE(res == ncclSuccess || res == ncclInProgress) << ncclGetErrorString(res);
      ASSERT_EQ(waitForComm(comm), ncclSuccess);
      EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
      HIP_EXPECT(hipFree(send));
      HIP_EXPECT(hipFree(recv));
      HIP_EXPECT(hipStreamDestroy(stream));
    }

    ProcessIsolatedTestRunner::TestConfig isolated(const std::string& name, bool rejectOneCall)
    {
      return ProcessIsolatedTestRunner::TestConfig(name, [rejectOneCall]() { singleRankNonBlocking(rejectOneCall); })
        .withNumGpus(1)
        .withEnvironment({{"NCCL_DEBUG", "WARN"}})
        .withTimeout(std::chrono::seconds(120));
    }
  } // namespace

  // NCCL #2459 / PR #2460: one call returning ncclInvalidArgument left a non-blocking comm unusable.
  TEST(NonBlocking, InvalidArgumentLeavesCommUsable)
  {
    RUN_ISOLATED_TESTS(isolated("InvalidArgumentLeavesCommUsable", true));
  }

  // Control: the same comm lifecycle without the rejected call.
  TEST(NonBlocking, SingleRankLifecycle)
  {
    RUN_ISOLATED_TESTS(isolated("SingleRankLifecycle", false));
  }
}
