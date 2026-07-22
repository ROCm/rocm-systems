/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // Coverage for the packed 2-wide fp8 reduction helpers added for gfx942/gfx950
  // (hminmax2/hminmax2_b in src/include/rccl_float8.h). These replace the
  // per-element widen/op/narrow scalar fallback that previously drove the large
  // private-segment (scratch) / register-spill use on the fp8 reduce kernels.
  //
  // MinMax keeps the fp8 values in range, so it validates the packed
  // convert -> v_pk_{min,max}_f16 -> convert path without the overflow concerns
  // of fp8 Sum/Prod. Runs out-of-place, non-graph, at both a large element count
  // (packed hot loop) and a small one.
  TEST(Fp8PackedReduce, AllReduceMinMax)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin, ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
