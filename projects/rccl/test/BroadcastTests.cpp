/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{
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

  TEST(Broadcast, SingleProcMemReg)
  {
    int hipRuntimeVer = 0;
    HIPCALL(hipRuntimeGetVersion(&hipRuntimeVer));
    if (hipRuntimeVer < 71260540) {
      GTEST_SKIP() << "Skipping SingleProcMemReg: HIP runtime version (" 
                   << hipRuntimeVer << ") is lower than 71260540";
    }
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclBfloat16, ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1,4314};
    std::vector<bool>           const inPlaceList     = {true,false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true,false};

    setenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE", "1", 1);
    setenv("NCCL_CUMEM_ENABLE","1",1);
    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList,true,MEM_ALLOC_SYMMETRIC_WIN);
    
    testBed.Finalize();
    unsetenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE");
    unsetenv("NCCL_CUMEM_ENABLE");
  }
}
