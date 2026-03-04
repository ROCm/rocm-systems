/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

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

  // Test ncclSum which is the most common reduction op in production.
  TEST(ReduceScatter, SumReduction)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Test in-place ReduceScatter with managed memory (hipMallocManaged).
  // Combines overlapping input/output buffers with automatic page migration.
  TEST(ReduceScatter, InPlaceManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {104857, 500};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Test in-place ReduceScatter with managed memory and HIP graph capture.
  // Validates the triple combination: overlapping buffers + page migration + graph replay.
  TEST(ReduceScatter, InPlaceManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {896};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // Test explicit buffer registration (ncclCommRegister/Deregister) with ReduceScatter.
  // Uses forked processes to mimic real distributed applications.
  TEST(ReduceScatter, UserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    size_t sendCount = nranks * count;
    std::vector<int> sendBuff(sendCount, 1);
    std::vector<int> recvBuff(count, 0);
    std::vector<int> expected(count, nranks);

    callCollectiveForked(nranks, ncclCollReduceScatter, sendBuff, recvBuff, expected);
  }

  // Test explicit buffer registration with ReduceScatter using managed memory.
  // Combines ncclCommRegister/Deregister with hipMallocManaged allocation.
  TEST(ReduceScatter, ManagedMemUserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    size_t sendCount = nranks * count;
    std::vector<int> sendBuff(sendCount, 1);
    std::vector<int> recvBuff(count, 0);
    std::vector<int> expected(count, nranks);
    const bool use_managed_mem = true;

    callCollectiveForked(nranks, ncclCollReduceScatter, sendBuff, recvBuff, expected, use_managed_mem);
  }
}
