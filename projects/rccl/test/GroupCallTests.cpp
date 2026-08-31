/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
#ifdef ENABLE_WARP_SPEED
  // Use a per-collective pattern offset so two same-shape AllGathers cannot pass
  // by accidentally sharing or exchanging their buffers.
  static ErrCode PrepareTaggedAllGatherData(CollectiveArgs& collArgs)
  {
    CHECK_CALL(CheckAllocation(collArgs));
    if (collArgs.totalRanks * collArgs.numInputElements != collArgs.numOutputElements)
    {
      TEST_ERROR("# of output elements must be total ranks * # input elements for AllGather");
      return TEST_FAIL;
    }

    size_t const numInputBytes  = collArgs.numInputElements * DataTypeToBytes(collArgs.dataType);
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);
    CHECK_CALL(collArgs.inputGpu.ClearGpuMem(numInputBytes));
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));

    PtrUnion result;
    CHECK_CALL(result.Attach(collArgs.expected.ptr));
    CHECK_CALL(result.ClearCpuMem(numOutputBytes));

    PtrUnion tempInputCpu;
    CHECK_CALL(tempInputCpu.Attach(collArgs.outputCpu.ptr));

    int const patternOffset = collArgs.options.inputConstantValue;
    for (int rank = 0; rank < collArgs.totalRanks; ++rank)
    {
      CHECK_CALL(tempInputCpu.FillPattern(collArgs.dataType, collArgs.numInputElements, patternOffset + rank, false));
      if (rank == collArgs.globalRank)
      {
        CHECK_HIP(hipMemcpy(collArgs.inputGpu.ptr, tempInputCpu.ptr, numInputBytes, hipMemcpyHostToDevice));
      }
      memcpy(result.I1 + rank * numInputBytes, tempInputCpu.ptr, numInputBytes);
    }
    return TEST_SUCCESS;
  }

  static void RequireGfx95EightRank(TestBed& testBed, int totalRanks)
  {
    if (!testBed.ev.isGfx95)
    {
      GTEST_SKIP() << "WarpSpeed multi-collective tests require gfx95 GPUs";
    }
    auto const& numGpusList = testBed.ev.GetNumGpusList();
    if (std::find(numGpusList.begin(), numGpusList.end(), totalRanks) == numGpusList.end())
    {
      GTEST_SKIP() << "Test requires exactly 8 ranks; adjust UT_MIN_GPUS/UT_MAX_GPUS if 8 GPUs are available";
    }
  }
#endif

  // Test identical collectives within the same group call
  TEST(GroupCall, Identical)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce, ncclCollAllReduce, ncclCollAllReduce};
    std::vector<ncclRedOp_t>    const testRedOps      = {ncclSum, ncclSum, ncclSum};
    std::vector<ncclDataType_t> const testDataTypes   = {ncclFloat, ncclFloat, ncclFloat};
    std::vector<int>            const numElements     = {1048576, 384 * 1024, 384};

    int                         const numCollPerGroup = numElements.size();
    bool                        const inPlace         = false;
    bool                        const useManagedMem   = false;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    std::vector<ncclRedOp_t> redOps;
    testBed.GetSupportedRedOps(redOps, testRedOps);
    if (redOps.empty()) {
      GTEST_SKIP() << "Skipping... test reduction operations excluded by UT_REDOPS.";
    }

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall Identical", isMultiProcess ? "MP" : "SP", totalRanks);

      // Set up the different collectives within the group
      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.redOp = redOps[collIdx];
        testBed.SetCollectiveArgs(funcTypes[collIdx],
                                  dataTypes[collIdx],
                                  numElements[collIdx],
                                  numElements[collIdx],
                                  options,
                                  collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Test different collectives within the same group call
  TEST(GroupCall, Different)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast,
                                                         ncclCollAllGather,
                                                         ncclCollReduceScatter,
                                                         ncclCollAllReduce,
                                                         ncclCollGather,
                                                         ncclCollScatter,
                                                         ncclCollAlltoAll};
    int                         const numCollPerGroup = funcTypes.size();
    int                         const numElements     = 1048576;
    bool                        const inPlace         = false;
    bool                        const useManagedMem   = false;

    OptionalColArgs options;
    options.redOp = ncclSum;
    options.root  = 0;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall Different", isMultiProcess ? "MP" : "SP", totalRanks);

      // Set up the different collectives within the group
      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        int numInputElements;
        int numOutputElements;
        CollectiveArgs::GetNumElementsForFuncType(funcTypes[collIdx],
                                                  numElements,
                                                  totalRanks,
                                                  &numInputElements,
                                                  &numOutputElements);

        testBed.SetCollectiveArgs(funcTypes[collIdx],
                                  ncclFloat,
                                  numInputElements,
                                  numOutputElements,
                                  options,
                                  collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Test identical collectives with different data type
  TEST(GroupCall, MixedDataType)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce, ncclCollAllReduce, ncclCollAllReduce};
    std::vector<ncclRedOp_t>    const testRedOps      = {ncclSum, ncclSum, ncclSum};
    std::vector<ncclDataType_t> const testDataTypes   = {ncclFloat16, ncclFloat32, ncclFloat64};
    std::vector<int>            const numElements     = {1048576, 384 * 1024, 384};

    int                         const numCollPerGroup = numElements.size();
    bool                        const inPlace         = false;
    bool                        const useManagedMem   = false;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    std::vector<ncclRedOp_t> redOps;
    testBed.GetSupportedRedOps(redOps, testRedOps);
    if (redOps.empty()) {
      GTEST_SKIP() << "Skipping... test reduction operations excluded by UT_REDOPS.";
    }

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MixedDataType", isMultiProcess ? "MP" : "SP", totalRanks);

      // Set up the different collectives within the group
      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.redOp = redOps[collIdx];
        testBed.SetCollectiveArgs(funcTypes[collIdx],
                                  dataTypes[collIdx],
                                  numElements[collIdx],
                                  numElements[collIdx],
                                  options,
                                  collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  TEST(GroupCall, Multistream)
  {
    TestBed testBed;

    // Configuration
    int  const  numElements        = 1048576;
    bool const  inPlace            = false;
    bool const  useManagedMem      = false;

    OptionalColArgs options;

    // This test runs multiple AllReduce collectives on different streams within the same group call
    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;

      for (int numCollPerGroup = 2; numCollPerGroup <= 6; numCollPerGroup += 2)
      {
        for (int numStreamsPerGroup = numCollPerGroup; numStreamsPerGroup >= 2; numStreamsPerGroup -= 3)
        {
          if (testBed.ev.showNames)
            TEST_INFO("%s %d-ranks Multistream %d-Group Calls across %d streams",
                 isMultiProcess ? "MP" : "SP", totalRanks, numCollPerGroup, numStreamsPerGroup);

          const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
          testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder),
                            numCollPerGroup, numStreamsPerGroup);

          // Set up each collective in group in different stream (modulo numStreamsPerGroup)
          options.redOp = ncclSum;
          for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
          {
            testBed.SetCollectiveArgs(ncclCollAllReduce, ncclFloat, numElements, numElements,
                                      options, collIdx, 0, -1, collIdx % numStreamsPerGroup);
          }

          testBed.AllocateMem(inPlace, useManagedMem);
          testBed.PrepareData();
          testBed.ExecuteCollectives();
          testBed.ValidateResults(isCorrect);
          testBed.DeallocateMem();
          testBed.DestroyComms();
        }
      }
    }
    testBed.Finalize();
  }

  TEST(GroupCall, MultiGroupCall)
  {
    TestBed testBed;

    // Configuration
    std::vector<std::vector<ncclFunc_t>> const groupCalls         = {{ncclCollAllReduce, ncclCollAllGather},
                                                                     {ncclCollAlltoAll, ncclCollGather},
                                                                     {ncclCollBroadcast, ncclCollReduceScatter}};
    std::vector<std::vector<int>>        const numElements        = {{1250, 1048576}, {384, 384 * 1024}, {1048576, 127}};
    std::vector<ncclDataType_t>          const testDataTypes      = {ncclFloat16, ncclFloat32, ncclBfloat16};
    std::vector<ncclRedOp_t>             const testRedOps         = {ncclSum, ncclProd, ncclMax};
    std::vector<int>                     const numCollsPerGroup   = {2, 2, 2};
    std::vector<int>                     const numStreamsPerGroup = {1, 1, 1};
    std::vector<bool>                    const useHipGraphList    = {true, false, true};
    bool                                 const inPlace            = false;
    bool                                 const useManagedMem      = false; 
    bool                                 const useBlocking        = true;
    int                                  const numGroupCalls      = groupCalls.size();
    int                                  const numIterations      = 10;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    std::vector<ncclRedOp_t> redOps;
    testBed.GetSupportedRedOps(redOps, testRedOps);
    if (redOps.empty()) {
      GTEST_SKIP() << "Skipping... test reduction operations excluded by UT_REDOPS.";
    }

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses     = isMultiProcess ? totalRanks : 1;

      // Initialize comms by specifying the # of group calls
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollsPerGroup, numStreamsPerGroup, numGroupCalls, useBlocking);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MultiGroupCall", isMultiProcess ? "MP" : "SP", totalRanks);
      
      for (int groupCallIdx = 0; groupCallIdx < groupCalls.size(); ++groupCallIdx)
      {
        std::vector<ncclFunc_t> funcTypes = groupCalls[groupCallIdx];
        OptionalColArgs options;
        options.redOp = redOps[groupCallIdx];
        options.root  = 0;

        for (int collIdx = 0; collIdx < numCollsPerGroup[groupCallIdx]; ++collIdx)
        {
          int numInputElements;
          int numOutputElements;
          CollectiveArgs::GetNumElementsForFuncType(funcTypes[collIdx],
                                                    numElements[groupCallIdx][collIdx],
                                                    totalRanks,
                                                    &numInputElements,
                                                    &numOutputElements);

          testBed.SetCollectiveArgs(funcTypes[collIdx],
                                    dataTypes[groupCallIdx],
                                    numInputElements,
                                    numOutputElements,
                                    options,
                                    collIdx,
                                    groupCallIdx);
        }

        testBed.AllocateMem(inPlace, useManagedMem, groupCallIdx);
        testBed.PrepareData(groupCallIdx);

        // Stream capture in advance for HIP graph enabled collective groups
        if (useHipGraphList[groupCallIdx])
        {
          testBed.ExecuteCollectives({}, groupCallIdx, useHipGraphList[groupCallIdx]);
        }
      }

      // Execute collectives based on groupIdx
      for (int i = 0; i < numIterations; ++i)
      {
        // Select a random group call
        int groupCallIdx = i % groupCalls.size();

        // Use graphs if enabled otherwise execute the collective
        if (useHipGraphList[groupCallIdx]) testBed.LaunchGraphs(groupCallIdx);
        else testBed.ExecuteCollectives({}, groupCallIdx);
        testBed.ValidateResults(isCorrect, groupCallIdx);
      }

      testBed.DeallocateMem();
      testBed.DestroyGraphs();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Group of broadcasts each rooted at a different rank. With >=2 distinct roots
  // and NCCL_ALLGATHERV_ENABLE=1 the task producer fuses them into a single
  // ncclFuncAllGatherV ring kernel. Sizes vary per root to exercise AllGatherV's
  // variable-length partitioning. AllGatherV fusion is disabled by default
  // (NCCL_ALLGATHERV_ENABLE=0); without the env var this test exercises the
  // individual-broadcast fallback path.
  TEST(GroupCall, MultiRootBroadcast)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Fusion requires >=2 distinct roots; skip single-GPU.
      if (totalRanks < 2) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      // One broadcast per rank-root => totalRanks distinct roots in one group.
      int const numCollPerGroup = totalRanks;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MultiRootBroadcast", isMultiProcess ? "MP" : "SP", totalRanks);

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.root = collIdx;                            // distinct root per broadcast
        size_t const numElements = 1048576 >> (collIdx % 4); // 1M, 512K, 256K, 128K, repeating
        testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                  numElements, numElements,
                                  options, collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Single broadcast in a group with AllGatherV enabled. BcastPeers==1 triggers the
  // downgrade path in ncclPrepareTasks (enqueue.cc) that converts the ncclTaskBcast
  // back into a plain ncclFuncBroadcast ncclTaskColl and inserts it into collSorter.
  // Validates that the downgrade produces correct data (not silently dropped).
  TEST(GroupCall, SingleRootBroadcastDowngrade)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (totalRanks < 2) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      // Exactly one broadcast with a non-zero root => BcastPeers==1 downgrade path.
      int const numCollPerGroup = 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall SingleRootBroadcastDowngrade", isMultiProcess ? "MP" : "SP", totalRanks);

      OptionalColArgs options;
      options.root = totalRanks - 1; // non-zero root stresses minBcastPeer indexing
      testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                1048576, 1048576,
                                options, /*collIdx=*/0);

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Two consecutive group calls on the same communicator with AllGatherV enabled.
  // The second call uses a different (non-zero) root to exercise reclaimPlannerState:
  // if minBcastPeer is not reset to INT_MAX after the first call, the second call
  // indexes into the wrong peer's bcastQueue and silently drops its task.
  TEST(GroupCall, MultiRootBroadcastConsecutive)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (totalRanks < 2) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      int const numCollPerGroup = totalRanks;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MultiRootBroadcastConsecutive", isMultiProcess ? "MP" : "SP", totalRanks);

      // First group call: roots 0..N-1
      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.root = collIdx;
        testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                  524288, 524288,
                                  options, collIdx);
      }
      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();

      // Second group call on the same communicator: roots in reverse order.
      // reclaimPlannerState must have reset minBcastPeer=INT_MAX so this call
      // correctly indexes peers[totalRanks-1] rather than peers[0].
      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.root = (numCollPerGroup - 1) - collIdx; // reverse root order
        testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                  262144, 262144,
                                  options, collIdx);
      }
      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();

      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Group call that mixes a broadcast (distinct root, AllGatherV path) with an AllReduce
  // (collSorter path) in the same ncclGroupStart/ncclGroupEnd. Exercises the case where
  // both collTaskAppend branches run in one planning cycle and the resulting plan must
  // launch both the AllGatherV ring kernel and the AllReduce kernel correctly.
  TEST(GroupCall, MixedBroadcastAndAllReduce)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (totalRanks < 2) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      // collIdx 0: Broadcast rooted at rank 1 (enters bcastQueue / AllGatherV path)
      // collIdx 1: Broadcast rooted at rank 0 (second distinct root -> BcastPeers==2, fuses)
      // collIdx 2: AllReduce (enters collSorter path)
      int const numCollPerGroup = 3;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MixedBroadcastAndAllReduce", isMultiProcess ? "MP" : "SP", totalRanks);

      OptionalColArgs bcastOpts0, bcastOpts1;
      bcastOpts0.root = 1;
      bcastOpts1.root = 0;
      testBed.SetCollectiveArgs(ncclCollBroadcast, dataType, 524288, 524288, bcastOpts0, /*collIdx=*/0);
      testBed.SetCollectiveArgs(ncclCollBroadcast, dataType, 262144, 262144, bcastOpts1, /*collIdx=*/1);
      testBed.SetCollectiveArgs(ncclCollAllReduce,  dataType, 262144, 262144, OptionalColArgs(), /*collIdx=*/2);

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Fused broadcast across pinned channel counts. The AllGatherV kernel launches
  // grid=nChannels and the scheduler charges its work-batch budget per channel
  // (nChannels * nWorks * sizeof(ncclDevWorkBcast)), so the channel count changes both
  // the launch geometry and how many device works one batch must hold. nChannels=1
  // covers the degenerate single-channel ring; larger counts cover multi-channel
  // partitioning. NCCL_MIN/MAX_NCHANNELS are consumed at communicator init, so they are
  // set before InitComms and cleared after the comms are destroyed.
  TEST(GroupCall, MultiRootBroadcastChannelCounts)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;
    // Two settings keep CI time bounded while still differing after the WarpSpeed multiplier
    // is divided out (gfx950 uses 4, so 1/32 land on 1 and 8 channels).
    std::vector<const char*> const channelList = {"1", "32"};

    bool isCorrect = true;
    for (auto channels : channelList)
    {
      setenv("NCCL_MIN_NCHANNELS", channels, 1);
      setenv("NCCL_MAX_NCHANNELS", channels, 1);

      for (int totalRanks : testBed.ev.GetNumGpusList())
      for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
      {
        // Fusion requires >=2 distinct roots; skip single-GPU.
        if (totalRanks < 2) continue;

        int const numProcesses = isMultiProcess ? totalRanks : 1;
        const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

        int const numCollPerGroup = totalRanks;
        testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

        if (testBed.ev.showNames)
          TEST_INFO("%s %d-ranks GroupCall MultiRootBroadcastChannelCounts nChannels=%s",
                    isMultiProcess ? "MP" : "SP", totalRanks, channels);

        for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
        {
          OptionalColArgs options;
          options.root = collIdx;
          size_t const numElements = 1048576 >> (collIdx % 4);
          testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                    numElements, numElements,
                                    options, collIdx);
        }

        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
        testBed.DestroyComms();
      }

      // Finalize inside the loop: pooled workers snapshot env at fork, so the
      // next channel count needs a freshly-forked pool to be seen.
      testBed.Finalize();
      unsetenv("NCCL_MIN_NCHANNELS");
      unsetenv("NCCL_MAX_NCHANNELS");
    }
  }

  // Fused broadcast with element counts that are not multiples of the protocol grain size
  // or the 16B pack size. The scheduler rounds chunkSize down to grainSize and slices each
  // task by ringDepth across channels, so unaligned counts are where a partitioning error
  // would drop or duplicate a tail. Counts include 1 element (smaller than one grain) and
  // primes just past a power of two; every receiver's buffer is validated.
  TEST(GroupCall, MultiRootBroadcastUnalignedSizes)
  {
    TestBed testBed;

    ncclDataType_t const dataType      = ncclFloat;
    bool           const inPlace       = false;
    bool           const useManagedMem = false;
    // Deliberately not multiples of 4 elements (16B for fp32), so each slice has a ragged tail:
    // below one grain, just past a pack, mid-size, and large enough to span multiple chunks.
    size_t const unalignedCounts[] = {1, 17, 1023, 1048577};
    int const numUnaligned = (int)(sizeof(unalignedCounts)/sizeof(unalignedCounts[0]));

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Fusion requires >=2 distinct roots; skip single-GPU.
      if (totalRanks < 2) continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      int const numCollPerGroup = totalRanks;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks GroupCall MultiRootBroadcastUnalignedSizes", isMultiProcess ? "MP" : "SP", totalRanks);

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.root = collIdx;                                          // distinct root per broadcast
        size_t const numElements = unalignedCounts[collIdx % numUnaligned];
        testBed.SetCollectiveArgs(ncclCollBroadcast, dataType,
                                  numElements, numElements,
                                  options, collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

#ifdef ENABLE_WARP_SPEED
  // Two AllGathers in one group with WarpSpeed forced on. Exercises packed-warp
  // batch indexing when collOpCount > 1 and distinct work lives on higher channels.
  TEST(GroupCall, WarpSpeedMultipleAllGatherEnabled)
  {
    TestBed testBed;

    int const totalRanks      = 8;
    int const numCollPerGroup = 2;
    size_t const numElements  = 4194304;
    bool const inPlace        = false;
    bool const useManagedMem  = false;

    RequireGfx95EightRank(testBed, totalRanks);

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclBfloat16});
    if (dataTypes.empty())
    {
      GTEST_SKIP() << "ncclBfloat16 excluded by UT_DATATYPES";
    }

    setenv("RCCL_WARP_SPEED_FORCE_ENABLE", "1", 1);

    bool isCorrect = true;
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      auto const& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s 8-ranks GroupCall WarpSpeedMultipleAllGatherEnabled", isMultiProcess ? "MP" : "SP");

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.inputConstantValue = collIdx * totalRanks;
        testBed.SetCollectiveArgs(ncclCollAllGather, dataTypes[0], numElements, totalRanks * numElements, options,
                                  collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData(/*groupId=*/-1, /*collId=*/-1, /*rank=*/-1, PrepareTaggedAllGatherData);
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }

    unsetenv("RCCL_WARP_SPEED_FORCE_ENABLE");
    testBed.Finalize();
  }

  // Two AllReduces in one group with WarpSpeed forced on.
  TEST(GroupCall, WarpSpeedMultipleAllReduceEnabled)
  {
    TestBed testBed;

    int const totalRanks      = 8;
    int const numCollPerGroup = 2;
    size_t const numElements  = 4194304;
    bool const inPlace        = false;
    bool const useManagedMem  = false;

    RequireGfx95EightRank(testBed, totalRanks);

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclFloat});
    if (dataTypes.empty())
    {
      GTEST_SKIP() << "ncclFloat excluded by UT_DATATYPES";
    }

    setenv("RCCL_WARP_SPEED_FORCE_ENABLE", "1", 1);

    bool isCorrect = true;
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      auto const& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s 8-ranks GroupCall WarpSpeedMultipleAllReduceEnabled", isMultiProcess ? "MP" : "SP");

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.redOp = ncclSum;
        options.inputConstantValue = collIdx + 1;
        testBed.SetCollectiveArgs(ncclCollAllReduce, dataTypes[0], numElements, numElements, options, collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }

    unsetenv("RCCL_WARP_SPEED_FORCE_ENABLE");
    testBed.Finalize();
  }

  // AllGather + AllReduce in one group with WarpSpeed forced on.
  TEST(GroupCall, WarpSpeedMixedAllGatherAllReduceEnabled)
  {
    TestBed testBed;

    int const totalRanks      = 8;
    int const numCollPerGroup = 2;
    size_t const agElements   = 4194304;
    size_t const arElements   = 4194304;
    bool const inPlace        = false;
    bool const useManagedMem  = false;

    RequireGfx95EightRank(testBed, totalRanks);

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclBfloat16});
    if (dataTypes.empty())
    {
      GTEST_SKIP() << "ncclBfloat16 excluded by UT_DATATYPES";
    }

    setenv("RCCL_WARP_SPEED_FORCE_ENABLE", "1", 1);

    bool isCorrect = true;
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      auto const& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("%s 8-ranks GroupCall WarpSpeedMixedAllGatherAllReduceEnabled", isMultiProcess ? "MP" : "SP");

      OptionalColArgs agOptions;
      agOptions.inputConstantValue = 0;
      testBed.SetCollectiveArgs(ncclCollAllGather, dataTypes[0], agElements, totalRanks * agElements, agOptions,
                                /*collIdx=*/0);

      OptionalColArgs arOptions;
      arOptions.redOp = ncclSum;
      arOptions.inputConstantValue = 3;
      testBed.SetCollectiveArgs(ncclCollAllReduce, dataTypes[0], arElements, arElements, arOptions, /*collIdx=*/1);

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData(/*groupId=*/-1, /*collId=*/0, /*rank=*/-1, PrepareTaggedAllGatherData);
      testBed.PrepareData(/*groupId=*/-1, /*collId=*/1, /*rank=*/-1, nullptr);
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }

    unsetenv("RCCL_WARP_SPEED_FORCE_ENABLE");
    testBed.Finalize();
  }
#endif
}
