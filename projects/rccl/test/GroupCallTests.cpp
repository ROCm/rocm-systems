/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glob.h>
#include <string>
#include <unistd.h>
#include <vector>

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

  static void EnableWarpSpeedGroupedTestEnv()
  {
    setenv("RCCL_WARP_SPEED_FORCE_ENABLE", "1", 1);
    // gfx950 WarpSpeed auto-selects >256 channels; validChannelsForWarpSpeed
    // then returns ncclInvalidArgument. Set before TestBed so pooled children inherit it.
    setenv("NCCL_MIN_NCHANNELS", "56", 1);
    setenv("NCCL_MAX_NCHANNELS", "56", 1);
  }

  struct WarpSpeedLaunchGeometry
  {
    int launches = 0;
    int gridX = 0;
    int nChannels = 0;
    int nWorkBatches = 0;
  };

  static void RemoveGlobbedFiles(std::string const& globPattern)
  {
    glob_t g{};
    if (glob(globPattern.c_str(), 0, nullptr, &g) == 0)
    {
      for (size_t i = 0; i < g.gl_pathc; ++i) std::remove(g.gl_pathv[i]);
    }
    globfree(&g);
  }

  // Parse Launch COLL kernel lines. Grouped WarpSpeed AllGathers must stay in one
  // plan: both colls chained on the launched channels (nWorkBatches ~ 2 * logical
  // channels), not split into one kernel per coll.
  static std::vector<WarpSpeedLaunchGeometry> ScrapeCollLaunches(std::string const& globPattern)
  {
    std::vector<WarpSpeedLaunchGeometry> out;
    glob_t g{};
    if (glob(globPattern.c_str(), 0, nullptr, &g) != 0)
    {
      globfree(&g);
      return out;
    }
    for (size_t i = 0; i < g.gl_pathc; ++i)
    {
      std::ifstream f(g.gl_pathv[i]);
      std::string line;
      while (std::getline(f, line))
      {
        auto const pos = line.find("Launch COLL kernel: gridDim.x=");
        if (pos == std::string::npos) continue;
        WarpSpeedLaunchGeometry geo;
        if (std::sscanf(line.c_str() + pos,
                        "Launch COLL kernel: gridDim.x=%d blockDim.x=%*d "
                        "(p2pnChannels=%*d, p2pnChannelsPerPeer=%*d, nChannels=%d, nWorkBatches=%d)",
                        &geo.gridX, &geo.nChannels, &geo.nWorkBatches) == 3)
        {
          geo.launches = 1;
          out.push_back(geo);
        }
      }
    }
    globfree(&g);
    return out;
  }

  static void ExpectWarpSpeedGroupedLaunchGeometry(std::string const& globPattern, int totalRanks)
  {
    auto const launches = ScrapeCollLaunches(globPattern);
    ASSERT_GE(static_cast<int>(launches.size()), totalRanks)
        << "expected at least one COLL launch per rank (Direct/P2P AllGather "
           "bypasses WarpSpeed and has no Launch COLL kernel line)";
    EXPECT_EQ(static_cast<int>(launches.size()) % totalRanks, 0)
        << "launch count " << launches.size()
        << " is not a multiple of ranks; grouped colls should stay in one kernel "
           "per rank per ExecuteCollectives()";
    int const warpsPerBlock = 4;
    for (auto const& geo : launches)
    {
      int const logicalChannels = geo.gridX * warpsPerBlock;
      // Bug packed each coll onto nChannels/2 (grid ~ nChannels/(2*warps)).
      // Per-task trafficPerChannel should use nearly the full WarpSpeed set.
      EXPECT_GE(logicalChannels, (9 * geo.nChannels) / 10)
          << "gridDim.x=" << geo.gridX << " warpsPerBlock=" << warpsPerBlock
          << " nChannels=" << geo.nChannels
          << ": WarpSpeed grouped colls should launch the full channel set";
      EXPECT_GE(geo.nWorkBatches, (9 * geo.nChannels) / 10)
          << "nWorkBatches=" << geo.nWorkBatches << " gridDim.x=" << geo.gridX
          << " nChannels=" << geo.nChannels
          << ": both grouped colls should be queued on the launched channels";
    }
    RemoveGlobbedFiles(globPattern);
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

  // Mix explicit P2P with AllToAll so a planner round can pair directions
  // originating from different APIs. The test intentionally checks completion
  // and data only; channel-count policy may change without weakening this
  // endpoint-agreement regression.
  TEST(GroupCall, MixedAllToAllAndP2pDirections)
  {
    TestBed testBed;
    if (testBed.ev.maxGpus < 4)
      GTEST_SKIP() << "Skipping... GroupCall.MixedAllToAllAndP2pDirections requires at least 4 GPUs";

    constexpr size_t p2pElements = 2 * 1024 * 1024; // 8 MiB: large enough to scale beyond one channel.
    constexpr size_t allToAllElementsPerPeer = 32 * 1024; // 128 KiB per peer: scales beyond one channel.
    constexpr int numCollPerGroup = 3;
    bool const inPlace = false;
    bool const useManagedMem = false;
    bool const useBlocking = false; // Let TestBed's timeout abort a channel-agreement regression instead of hanging.
    bool isCorrect = true;

    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (totalRanks < 4) continue;
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder),
                        numCollPerGroup, 1, 1, useBlocking);

      auto setP2p = [&](int rank, int collId, ncclFunc_t func, int peer) {
        OptionalColArgs options;
        options.root = peer;
        testBed.SetCollectiveArgs(func, ncclFloat32, p2pElements, p2pElements, options, collId, 0, rank);
      };

      // Before AllToAll, queue 0->1, 1->2, and 0->2. For P2P round d=1,
      // rank 0 pairs an AllToAll recv with its explicit send to rank 1,
      // rank 1 pairs explicit recv/send, and rank 2 pairs its explicit recv
      // with an AllToAll send. Remaining ranks use self P2P to fill both slots
      // without changing the inter-rank peer queues.
      setP2p(0, 0, ncclCollSend, 1);
      setP2p(0, 1, ncclCollSend, 2);
      setP2p(1, 0, ncclCollRecv, 0);
      setP2p(1, 1, ncclCollSend, 2);
      setP2p(2, 0, ncclCollRecv, 1);
      setP2p(2, 1, ncclCollRecv, 0);
      for (int rank = 3; rank < totalRanks; ++rank) {
        setP2p(rank, 0, ncclCollSend, rank);
        setP2p(rank, 1, ncclCollRecv, rank);
      }

      size_t const allToAllElements = allToAllElementsPerPeer * totalRanks;
      testBed.SetCollectiveArgs(ncclCollAlltoAll, ncclFloat32, allToAllElements, allToAllElements,
                                OptionalColArgs(), 2);

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
    EnableWarpSpeedGroupedTestEnv();
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

    testBed.Finalize();
  }

  // Same grouped AllGather as WarpSpeedMultipleAllGatherEnabled, plus launch-geometry
  // checks. A packing change that splits the group into one kernel per coll still
  // verifies bytes, so correctness tests would miss it.
  TEST(GroupCall, WarpSpeedMultipleAllGatherLaunchGeometry)
  {
    std::string const debugPrefix =
        std::string("/tmp/rccl_ws_grouped_ag_") + std::to_string(getpid());
    std::string const debugFile = debugPrefix + ".%p.log";
    std::string const debugGlob = debugPrefix + ".*";
    RemoveGlobbedFiles(debugGlob);
    EnableWarpSpeedGroupedTestEnv();
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "COLL", 1);
    setenv("NCCL_DEBUG_FILE", debugFile.c_str(), 1);

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

    bool isCorrect = true;
    bool scraped = false;
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (isMultiProcess) continue;
      scraped = true;
      int const numProcesses = 1;
      auto const& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("SP 8-ranks GroupCall WarpSpeedMultipleAllGatherLaunchGeometry");

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

    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    testBed.Finalize();

    if (!scraped)
    {
      GTEST_SKIP() << "Launch-geometry check is single-process only";
    }

    ExpectWarpSpeedGroupedLaunchGeometry(debugGlob, totalRanks);
  }

  // Two AllGathers of different sizes in one group. The trafficPerChannel sum
  // used the combined byte count, so a small coll could shrink packing of a
  // large one (and vice versa). Per-task packing must still fill the channel set
  // and keep both colls in one kernel.
  TEST(GroupCall, WarpSpeedMultipleAllGatherUnequalSizes)
  {
    // NCCL_DEBUG_FILE is cached on first use in this process (LaunchGeometry
    // already set it). Keep the same prefix so scrape sees this test's launches.
    std::string const debugPrefix =
        std::string("/tmp/rccl_ws_grouped_ag_") + std::to_string(getpid());
    std::string const debugFile = debugPrefix + ".%p.log";
    std::string const debugGlob = debugPrefix + ".*";
    RemoveGlobbedFiles(debugGlob);
    EnableWarpSpeedGroupedTestEnv();
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "COLL", 1);
    setenv("NCCL_DEBUG_FILE", debugFile.c_str(), 1);

    TestBed testBed;

    int const totalRanks            = 8;
    int const numCollPerGroup       = 2;
    size_t const numElementsSmall   = 1048576;  // 2 MiB BF16 send (keep Ring/WarpSpeed, not Direct)
    size_t const numElementsLarge   = 4194304;  // 8 MiB BF16 send (ticket size)
    size_t const numElements[2]     = {numElementsSmall, numElementsLarge};
    bool const inPlace              = false;
    bool const useManagedMem        = false;

    RequireGfx95EightRank(testBed, totalRanks);

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclBfloat16});
    if (dataTypes.empty())
    {
      GTEST_SKIP() << "ncclBfloat16 excluded by UT_DATATYPES";
    }

    bool isCorrect = true;
    bool scraped   = false;
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      if (isMultiProcess) continue;
      scraped                    = true;
      int const numProcesses     = 1;
      auto const& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        TEST_INFO("SP 8-ranks GroupCall WarpSpeedMultipleAllGatherUnequalSizes");

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        OptionalColArgs options;
        options.inputConstantValue = collIdx * totalRanks;
        testBed.SetCollectiveArgs(ncclCollAllGather, dataTypes[0], numElements[collIdx],
                                  totalRanks * numElements[collIdx], options, collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData(/*groupId=*/-1, /*collId=*/-1, /*rank=*/-1, PrepareTaggedAllGatherData);
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }

    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    testBed.Finalize();

    if (!scraped)
    {
      GTEST_SKIP() << "Launch-geometry check is single-process only";
    }

    ExpectWarpSpeedGroupedLaunchGeometry(debugGlob, totalRanks);
  }

  // Two AllReduces in one group with WarpSpeed forced on.
  TEST(GroupCall, WarpSpeedMultipleAllReduceEnabled)
  {
    EnableWarpSpeedGroupedTestEnv();
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

    testBed.Finalize();
  }

  // AllGather + AllReduce in one group with WarpSpeed forced on.
  TEST(GroupCall, WarpSpeedMixedAllGatherAllReduceEnabled)
  {
    EnableWarpSpeedGroupedTestEnv();
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

    testBed.Finalize();
  }
#endif
}
