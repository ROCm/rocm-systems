/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "plugin/nccl_tuner.h"

namespace RcclUnitTesting
{
  // Runs AllReduce at two message sizes for every unroll factor (indices 0-5).
  // This binary is built alongside --all-unrolls and is meant to be run
  // against a library built with RCCL_BUILD_ALL_UNROLLS=ON.
  TEST(AllReduce, UnrollFactorSweep)
  {
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

#ifndef RCCL_BUILD_ALL_UNROLLS
    GTEST_SKIP() << "Skipped: requires --all-unrolls build (RCCL_BUILD_ALL_UNROLLS=ON)";
#else
    for (int idx = NCCL_UNROLL_1; idx <= NCCL_UNROLL_32; ++idx) {
      char idxStr[4];
      snprintf(idxStr, sizeof(idxStr), "%d", idx);
      setenv("RCCL_UNROLL_FACTOR", idxStr, 1);

      TestBed testBed;
      testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                             inPlaceList, managedMemList, useHipGraphList);
      testBed.Finalize();

      unsetenv("RCCL_UNROLL_FACTOR");
    }
#endif
  }
}
