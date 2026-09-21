/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "SingleProcMemRegTestUtils.hpp"
#include "StandaloneUtils.hpp"

#include <cstdio>
#include <string>
#include <unistd.h>

namespace RcclUnitTesting
{
  // Keyed by the test process pid: every TestBed worker is its direct child.
  static std::string ExplicitPrepMarker(pid_t testPid)
  {
    return "/dev/shm/rccl_ut_explicit_prep_" + std::to_string(testPid);
  }

  // Runs in the worker. The marker proves the decoded offset reached this function rather
  // than the default dispatcher, which also ends in DefaultPrepData_Broadcast.
  static ErrCode MarkedPrepData_Broadcast(CollectiveArgs& collArgs)
  {
    if (FILE* f = fopen(ExplicitPrepMarker(getppid()).c_str(), "w")) fclose(f);
    return DefaultPrepData_Broadcast(collArgs);
  }

  // Owned by the test process: removes any stale marker up front and the marker on every exit.
  struct ExplicitPrepMarkerFile
  {
    std::string const path = ExplicitPrepMarker(getpid());
    ExplicitPrepMarkerFile()  { std::remove(path.c_str()); }
    ~ExplicitPrepMarkerFile() { std::remove(path.c_str()); }
  };

  TEST(Broadcast, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
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

  TEST(Broadcast, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
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

  TEST(Broadcast, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = {104857, 264};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Broadcast, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = {958};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Broadcast, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
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

  TEST(Broadcast, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
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

  // An explicit prepDataFunc crosses the parent/worker pipe. Workers are exec'd, so a
  // raw parent code address would be invalid in them under ASLR.
  TEST(Broadcast, ExplicitPrepareDataFunc)
  {
    TestBed testBed;
    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, {ncclFloat32, ncclInt32});
    if (dataTypes.empty())
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";

    EXPECT_EQ(CollFuncPtrFromOffset(CollFuncPtrToOffset(nullptr)), &DefaultPrepareDataFunc);
    EXPECT_EQ(CollFuncPtrFromOffset(CollFuncPtrToOffset(&MarkedPrepData_Broadcast)),
              &MarkedPrepData_Broadcast);

    ExplicitPrepMarkerFile const markerFile;
    std::string const& marker = markerFile.path;

    int const totalRanks = testBed.ev.maxGpus;
    OptionalColArgs options;
    options.root = totalRanks - 1;

    bool isCorrect = true;
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks,
                                                  testBed.ev.GetGpuPriorityOrder()));
      testBed.SetCollectiveArgs(ncclCollBroadcast, dataTypes[0], 4096, 4096, options);
      testBed.AllocateMem();
      testBed.PrepareData(-1, -1, -1, MarkedPrepData_Broadcast);
      EXPECT_EQ(access(marker.c_str(), F_OK), 0)
        << "Workers did not run the explicit prepDataFunc (" << (isMultiProcess ? "MP" : "SP") << ")";
      std::remove(marker.c_str());
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();
    }
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  TEST(Broadcast, SingleProcMemReg)
  {
    SingleProcMemRegTestConfig config;
    config.mode = SingleProcMemRegMode::Enabled;
    config.funcTypes = {ncclCollBroadcast};
    config.dataTypes = {ncclUint8, ncclBfloat16, ncclUint32, ncclUint64};
    config.redOps = {ncclSum};
    config.roots = {0};
    config.numElements = {1, 4314};
    config.inPlaceList = {true, false};
    config.useHipGraphList = {true, false};
    RunSingleProcMemRegTest(config);
  }

  TEST(Broadcast, SingleProcMemRegDisabledSmoke)
  {
    RunSingleProcMemRegDisabledSmoke(ncclCollBroadcast, ncclUint32, true);
  }
}
