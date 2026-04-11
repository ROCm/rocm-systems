/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

#ifdef RCCL_ALLREDUCE_WITH_BIAS

// AllReduceBias tests mirror the AllReduce test structure exactly.
// The only difference is options.useBias = true, which causes RunSimpleSweep
// to set biasNumElements on each collective call.
// All tests require gfx942 or gfx950 architecture.

namespace RcclUnitTesting
{
#define BIAS_SKIP_CHECK() \
  do { \
    if (!testBed.ev.isGfx94 && !testBed.ev.isGfx95) \
      GTEST_SKIP() << "Requires gfx942 or gfx950 architecture."; \
  } while (0)

  namespace BiasTestConstants
  {
    // Default datatypes exercised by bias tests
    std::vector<ncclDataType_t> const DEFAULT_DATATYPES = {ncclBfloat16, ncclFloat32, ncclFloat8e5m2};

    // Element counts for different test cases
    std::vector<int> const ELEM_COUNTS_LARGE         = {393216, 384};
    std::vector<int> const ELEM_COUNTS_MEDIUM        = {12888};
    std::vector<int> const ELEM_COUNTS_SMALL         = {384};
    std::vector<int> const ELEM_COUNTS_VARYING       = {393216, 12888, 384};
    std::vector<int> const ELEM_COUNTS_MANAGED       = {2500};
    std::vector<int> const ELEM_COUNTS_MANAGED_GRAPH = {4314};

    // Shared options for all bias tests: only useBias needs to be set;
    // biasConstantValue and inputConstantValue default to -1 (incremental/rank-based).
    inline OptionalColArgs biasOptions()
    {
      OptionalColArgs o;
      o.useBias = true;
      return o;
    }
  } // namespace BiasTestConstants

  TEST(AllReduceBias, OutOfPlace)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_LARGE);
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

  TEST(AllReduceBias, OutOfPlaceGraph)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_MEDIUM);
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

  TEST(AllReduceBias, InPlace)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_SMALL);
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

  TEST(AllReduceBias, InPlaceGraph)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_VARYING);
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

  TEST(AllReduceBias, ManagedMem)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_MANAGED);
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

  TEST(AllReduceBias, ManagedMemGraph)
  {
    TestBed testBed;
    BIAS_SKIP_CHECK();

    using namespace BiasTestConstants;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes(DEFAULT_DATATYPES);
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements(ELEM_COUNTS_MANAGED_GRAPH);
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, biasOptions());
    testBed.Finalize();
  }

#undef BIAS_SKIP_CHECK
}

#endif // RCCL_ALLREDUCE_WITH_BIAS
