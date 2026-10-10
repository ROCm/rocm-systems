/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

namespace RcclUnitTesting
{
  TEST(AllGather, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFloat64, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {586};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {104857, 264};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {958};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1039203, 2500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {896};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, UserBufferRegistration)
  {          
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(nranks*count, 0);

    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];

    callCollectiveForked(nranks, ncclCollAllGather, sendBuff, recvBuff, expected);
  }

  TEST(AllGather, ManagedMemUserBufferRegistration)
  {          
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(nranks*count, 0);
    const bool use_managed_mem = true;
    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];

    callCollectiveForked(nranks, ncclCollAllGather, sendBuff, recvBuff, expected, use_managed_mem);
  }

  // Out-of-place AllGather on ncclCommRegister'd buffers driven through TestBed. Input and
  // output sizes differ, so each registration must use its own buffer's size.
  TEST(AllGather, UserBufferRegistrationTestBed)
  {
    TestBed testBed;
    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclInt32, ncclFloat32});
    if (dataTypes.empty())
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";

    std::vector<int> const numElements    = {1048576, 1024};
    bool             const inPlace        = false;
    bool             const useManagedMem  = false;
    bool             const userRegistered = true;
    int              const totalRanks     = testBed.ev.maxGpus;

    bool isCorrect = true;
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks,
                                                  testBed.ev.GetGpuPriorityOrder()));

      for (size_t dtIdx = 0; dtIdx < dataTypes.size() && isCorrect; ++dtIdx)
      for (size_t neIdx = 0; neIdx < numElements.size() && isCorrect; ++neIdx)
      {
        int numInputElements, numOutputElements;
        CollectiveArgs::GetNumElementsForFuncType(ncclCollAllGather, numElements[neIdx], totalRanks,
                                                  &numInputElements, &numOutputElements);
        if (testBed.ev.showNames)
          TEST_INFO("%s AllGather UserBufferRegistration %s [%d elements]",
                    isMultiProcess ? "MP" : "SP", ncclDataTypeNames[dataTypes[dtIdx]],
                    numInputElements);
        testBed.SetCollectiveArgs(ncclCollAllGather, dataTypes[dtIdx],
                                  numInputElements, numOutputElements);
        testBed.AllocateMem(inPlace, useManagedMem, -1, -1, -1, userRegistered);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    }
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }
}
