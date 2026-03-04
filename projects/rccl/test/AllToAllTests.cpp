/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 // Note: InPlace is not supported for All-To-All

#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

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

  // Test AllToAll with integer types ncclInt32/Int8/Int64.
  TEST(AlltoAll, IntegerTypes)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAlltoAll};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8, ncclInt64};
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

  // Test explicit buffer registration (ncclCommRegister/Deregister) with AllToAll.
  // Uses forked processes to mimic real distributed applications.
  TEST(AlltoAll, UserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    size_t totalCount = nranks * count;
    std::vector<int> sendBuff(totalCount, 1);
    std::vector<int> recvBuff(totalCount, 0);
    std::vector<int> expected(totalCount, 1);

    callCollectiveForked(nranks, ncclCollAlltoAll, sendBuff, recvBuff, expected);
  }

  // Test explicit buffer registration with AllToAll using managed memory.
  // Combines ncclCommRegister/Deregister with hipMallocManaged allocation.
  TEST(AlltoAll, ManagedMemUserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    size_t totalCount = nranks * count;
    std::vector<int> sendBuff(totalCount, 1);
    std::vector<int> recvBuff(totalCount, 0);
    std::vector<int> expected(totalCount, 1);
    const bool use_managed_mem = true;

    callCollectiveForked(nranks, ncclCollAlltoAll, sendBuff, recvBuff, expected, use_managed_mem);
  }
}