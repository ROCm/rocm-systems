/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // FP8 reductions validated against one expected result from the verifiable
  // generator; see common/VerifiableFp8.hpp. The reference does not depend
  // on the algorithm, so symmetric sweeps rely on RunSimpleSweep skipping rank
  // configurations whose comms report no symmetric kernel support.
  static void RunFp8VerifiableSweep(ncclFunc_t const funcType,
                                    std::vector<ncclRedOp_t> const& redOps,
                                    MemAllocType const memAllocType)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {funcType};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 12345, 7};
    std::vector<bool>           const inPlaceList     = {false, true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, true, memAllocType);
    testBed.Finalize();
  }

  static std::vector<ncclRedOp_t> const kFp8Ops = {ncclSum, ncclProd, ncclMax, ncclMin};

  TEST(Fp8Verifiable, AllReduce)              { RunFp8VerifiableSweep(ncclCollAllReduce,     kFp8Ops,   MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, Reduce)                 { RunFp8VerifiableSweep(ncclCollReduce,        kFp8Ops,   MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, ReduceScatter)          { RunFp8VerifiableSweep(ncclCollReduceScatter, kFp8Ops,   MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, AllReduceSymmetric)     { RunFp8VerifiableSweep(ncclCollAllReduce,     kFp8Ops,   MEM_ALLOC_SYMMETRIC_WIN); }
  TEST(Fp8Verifiable, ReduceScatterSymmetric) { RunFp8VerifiableSweep(ncclCollReduceScatter, kFp8Ops,   MEM_ALLOC_SYMMETRIC_WIN); }

  // Disabled: on gfx942 FP8 ncclAvg returns half the average, because enqueue.cc
  // encodes the 1/nRanks PreMulSum scalar as OCP FP8 on the host while the device
  // reads it as fnuz. With one rank it aborts instead: the host looks up
  // oneRankReduce<FuncPreMulSum<__hip_fp8_e4m3>>, but the device has only the fnuz one.
  TEST(Fp8Verifiable, DISABLED_AvgAllReduce)              { RunFp8VerifiableSweep(ncclCollAllReduce,     {ncclAvg}, MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, DISABLED_AvgReduce)                 { RunFp8VerifiableSweep(ncclCollReduce,        {ncclAvg}, MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, DISABLED_AvgReduceScatter)          { RunFp8VerifiableSweep(ncclCollReduceScatter, {ncclAvg}, MEM_ALLOC_HIP); }
  TEST(Fp8Verifiable, DISABLED_AvgAllReduceSymmetric)     { RunFp8VerifiableSweep(ncclCollAllReduce,     {ncclAvg}, MEM_ALLOC_SYMMETRIC_WIN); }
  TEST(Fp8Verifiable, DISABLED_AvgReduceScatterSymmetric) { RunFp8VerifiableSweep(ncclCollReduceScatter, {ncclAvg}, MEM_ALLOC_SYMMETRIC_WIN); }
}
