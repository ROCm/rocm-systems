/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef VERIFIABLE_FP8_HPP
#define VERIFIABLE_FP8_HPP
#include <cstdint>
#include <hip/hip_runtime.h>
#include "rccl/rccl.h"

namespace RcclUnitTesting
{
  // Seeded FP8 reduction data whose result is exact in any accumulation order and precision,
  // so a single expected value applies to every algorithm. Every rank's input differs.
  // Supports ncclFloat8e4m3 / ncclFloat8e5m2 with ncclSum, ncclProd, ncclMax, ncclMin and ncclAvg.
  // Values use the device's native FP8 encoding (fnuz or OCP), matching RCCL's collectives.
  // eltIx0 is the index of elts[0] within the full reduced vector (e.g. for ReduceScatter).

  // Fills this rank's contribution to the reduction
  hipError_t VerifiableFp8PrepareInput(void* elts, intptr_t eltN, ncclDataType_t dataType,
                                       ncclRedOp_t redOp, int rankN, int rankMe,
                                       uint64_t seed, intptr_t eltIx0, hipStream_t stream);

  // Writes the expected reduction result
  hipError_t VerifiableFp8PrepareExpected(void* elts, intptr_t eltN, ncclDataType_t dataType,
                                          ncclRedOp_t redOp, int rankN,
                                          uint64_t seed, intptr_t eltIx0, hipStream_t stream);

  // Counts result elements that differ from the expected result into *badEltN,
  // which must be host-pinned memory (valid after the stream synchronizes).
  // Exact, except Avg allows 2 ULP for its rounded 1/rankN pre-multiply.
  hipError_t VerifiableFp8Verify(void const* results, intptr_t eltN, ncclDataType_t dataType,
                                 ncclRedOp_t redOp, int rankN, uint64_t seed, intptr_t eltIx0,
                                 int64_t* badEltN, hipStream_t stream);
}

#endif // VERIFIABLE_FP8_HPP
