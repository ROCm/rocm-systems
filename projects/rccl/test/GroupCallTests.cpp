/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // Test identical collectives within the same group call
  TEST(GroupCall, Identical)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce, ncclCollAllReduce, ncclCollAllReduce};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclSum, ncclSum};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat, ncclFloat, ncclFloat};
    std::vector<int>            const numElements     = {1048576, 384 * 1024, 384};

    int                         const numCollPerGroup = numElements.size();
    bool                        const inPlace         = false;
    bool                        const useManagedMem   = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        INFO("%s %d-ranks GroupCall Identical\n", isMultiProcess ? "MP" : "SP", totalRanks);

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
                                                         ncclCollAllToAll};
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
        INFO("%s %d-ranks GroupCall Different\n", isMultiProcess ? "MP" : "SP", totalRanks);

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
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclSum, ncclSum};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32, ncclFloat64};
    std::vector<int>            const numElements     = {1048576, 384 * 1024, 384};

    int                         const numCollPerGroup = numElements.size();
    bool                        const inPlace         = false;
    bool                        const useManagedMem   = false;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      // Test either single process all GPUs, or 1 process per GPU
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollPerGroup);

      if (testBed.ev.showNames)
        INFO("%s %d-ranks GroupCall MixedDataType\n", isMultiProcess ? "MP" : "SP", totalRanks);

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
            INFO("%s %d-ranks Multistream %d-Group Calls across %d streams\n",
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
                                                                     {ncclCollAllToAll, ncclCollGather},
                                                                     {ncclCollBroadcast, ncclCollReduceScatter}};
    std::vector<std::vector<int>>        const numElements        = {{1250, 1048576}, {384, 384 * 1024}, {1048576, 127}};
    std::vector<ncclDataType_t>          const dataTypes          = {ncclFloat16, ncclFloat32, ncclBfloat16};
    std::vector<ncclRedOp_t>             const redops             = {ncclSum, ncclProd, ncclMax};
    std::vector<int>                     const numCollsPerGroup   = {2, 2, 2};
    std::vector<int>                     const numStreamsPerGroup = {1, 1, 1};
    std::vector<bool>                    const useHipGraphList    = {true, false, true};
    bool                                 const inPlace            = false;
    bool                                 const useManagedMem      = false; 
    bool                                 const useBlocking        = true;
    int                                  const numGroupCalls      = groupCalls.size();
    int                                  const numIterations      = 10;

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses     = isMultiProcess ? totalRanks : 1;

      // Initialize comms by specifying the # of group calls
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), numCollsPerGroup, numStreamsPerGroup, numGroupCalls, useBlocking);

      if (testBed.ev.showNames)
        INFO("%s %d-ranks GroupCall MultiGroupCall\n", isMultiProcess ? "MP" : "SP", totalRanks);
      
      for (int groupCallIdx = 0; groupCallIdx < groupCalls.size(); ++groupCallIdx)
      {
        std::vector<ncclFunc_t> funcTypes = groupCalls[groupCallIdx];
        OptionalColArgs options;
        options.redOp = redops[groupCallIdx];
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
}
