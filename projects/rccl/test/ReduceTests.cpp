/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Reduce, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat32});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216, 384});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat16, ncclFloat64, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclInt32, ncclInt8});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = testBed.ev.GetElements({384});
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216});
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclUint64});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({3524082, 2500});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat64, ncclBfloat16});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({4314});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
