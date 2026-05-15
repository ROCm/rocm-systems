/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

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
          INFO("%s %d-ranks Non-Blocking %s\n",
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
}
