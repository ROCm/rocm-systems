/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "SingleProcMemRegTestUtils.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{
  TEST(ReduceScatter, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const roots           = {0, 1};
    std::vector<int>            const numElements     = {542357};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclFloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {246};;
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt64, ncclUint8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclAvg};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclAvg};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {6485423};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // In-place ReduceScatter reads the whole input allocation, so registration must
  // cover the base buffer rather than this rank's output slice.
  TEST(ReduceScatter, InPlaceUserBufferRegistrationTestBed)
  {
    TestBed testBed;
    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclInt32, ncclFloat32});
    if (dataTypes.empty())
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";

    std::vector<int> const numElements    = {1048576, 1024};
    bool             const inPlace        = true;
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
        CollectiveArgs::GetNumElementsForFuncType(ncclCollReduceScatter, numElements[neIdx], totalRanks,
                                                  &numInputElements, &numOutputElements);
        if (testBed.ev.showNames)
          TEST_INFO("%s ReduceScatter InPlaceUserBufferRegistration %s [%d elements]",
                    isMultiProcess ? "MP" : "SP", ncclDataTypeNames[dataTypes[dtIdx]],
                    numOutputElements);
        testBed.SetCollectiveArgs(ncclCollReduceScatter, dataTypes[dtIdx],
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

  TEST(ReduceScatter, SingleProcMemReg)
  {
    SingleProcMemRegTestConfig config;
    config.mode = SingleProcMemRegMode::Enabled;
    config.funcTypes = {ncclCollReduceScatter};
    config.dataTypes = {ncclFloat64, ncclFloat32, ncclFloat16,
                        ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
    config.redOps = {ncclSum};
    config.roots = {0};
    config.numElements = {1, 3, 7, 4314, 5003, 1048575, 1048576};
    config.inPlaceList = {true, false};
    config.useHipGraphList = {true, false};
    RunSingleProcMemRegTest(config);
  }

  TEST(ReduceScatter, SingleProcMemRegDisabled)
  {
    RunSingleProcMemRegDisabledTest(ncclCollReduceScatter, ncclFloat32, true);
  }
}
