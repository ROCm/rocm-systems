/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(ReduceScatter, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat32});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclMax});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216, 384});
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
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat64, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclMax});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({1048576});
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
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclInt32});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclProd});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({542357});
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
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclUint8, ncclFloat16});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclMin});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({246});
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
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclInt64, ncclUint8});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclAvg});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({1024});
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
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclUint32, ncclUint64});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclAvg});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({6485423});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
