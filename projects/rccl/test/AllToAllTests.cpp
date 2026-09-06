/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 // Note: InPlace is not supported for All-To-All

#include "TestBed.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{
  TEST(AlltoAll, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AlltoAll, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {5685};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AlltoAll, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {384 * 1024, 1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AlltoAll, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

    TEST(AlltoAll, Channels)
  {
    TestBed testBed;
    if(testBed.ev.maxGpus >= 8) {
      if(testBed.ev.isGfx94) {
        // Configuration
        std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
        std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16};
        std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
        std::vector<int>            const roots           = {0};
        std::vector<int>            const numElements     = {64 * 1024 * 1024, 1024};
        std::vector<bool>           const inPlaceList     = {false};
        std::vector<bool>           const managedMemList  = {false};
        std::vector<bool>           const useHipGraphList = {false, true};
        std::vector<const char *>   const channelList     = {"112"}; 
        bool                        const enableSweep     = false;
        for (auto channel : channelList) {
          setenv("NCCL_MIN_NCHANNELS", channel, 1);
          testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                inPlaceList, managedMemList, useHipGraphList, enableSweep);
          testBed.Finalize();
          unsetenv("NCCL_MIN_NCHANNELS");
        }
      }
    }
  }


  TEST(AlltoAll, SingleProcMemReg)
  {
    int hipRuntimeVer = 0;
    HIPCALL(hipRuntimeGetVersion(&hipRuntimeVer));
    if (hipRuntimeVer < 71260540) {
      GTEST_SKIP() << "Skipping SingleProcMemReg: HIP runtime version (" 
                   << hipRuntimeVer << ") is lower than 71260540";
    }

    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclBfloat16, ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1,4096,4314};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    setenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE", "1", 1);
    setenv("NCCL_CUMEM_ENABLE","1",1);
    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList,true,MEM_ALLOC_SYMMETRIC_WIN);
    
    testBed.Finalize();
    unsetenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE");
    unsetenv("NCCL_CUMEM_ENABLE");
  }

  TEST(AlltoAll, SingleProcMemRegGraph)
  {
    int hipRuntimeVer = 0;
    HIPCALL(hipRuntimeGetVersion(&hipRuntimeVer));
    if (hipRuntimeVer < 71260540) {
      GTEST_SKIP() << "Skipping SingleProcMemReg: HIP runtime version (" 
                   << hipRuntimeVer << ") is lower than 71260540";
    }
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8,ncclBfloat16,ncclUint32,ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1,4096,5003,5012};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    setenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE", "1", 1);
    setenv("NCCL_CUMEM_ENABLE","1",1);
    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList,true,MEM_ALLOC_SYMMETRIC_WIN);
    
    testBed.Finalize();
    unsetenv("NCCL_SINGLE_PROC_MEM_REG_ENABLE");
    unsetenv("NCCL_CUMEM_ENABLE");
  }
}