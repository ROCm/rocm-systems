/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "VerifiableData.hpp"
#include "CollectiveArgs.hpp"
#include "VerifiableFp8.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

// FP8 reduction validation in the TestBed:
// - PrepData: each rank generates its own input on the GPU from a seed shared by all ranks
//   (VerifiableFp8PrepareInput); no rank needs another rank's data and no host reference is built.
// - ValidateResults: each rank checks its output on the GPU against the expected values
//   regenerated from the same seed (VerifiableFp8Verify). The result is exact in any
//   accumulation order, so one reference covers every algorithm and protocol.
// - ReduceScatter checks its slice at offset globalRank * numOutputElements.
// - UT_VERIFIABLE_FAULT=1 flips one output byte, as a negative control.
namespace RcclUnitTesting
{
  // Must be identical on every rank, and independent of the element count so
  // that RunSimpleSweep can reuse a prepared input for smaller sub-cases.
  static uint64_t VerifiableSeed(CollectiveArgs const& collArgs)
  {
    char const* env = getenv("UT_VERIFIABLE_SEED");
    uint64_t const base = env ? strtoull(env, nullptr, 0) : 0x5eedULL;
    return base ^ ((uint64_t)collArgs.funcType << 40) ^ ((uint64_t)collArgs.dataType << 32)
                ^ ((uint64_t)collArgs.options.redOp << 24) ^ (uint64_t)collArgs.totalRanks;
  }

  // Index of this rank's first output element within the reduced vector
  static intptr_t OutputEltIx0(CollectiveArgs const& collArgs)
  {
    return collArgs.funcType == ncclCollReduceScatter
      ? (intptr_t)collArgs.globalRank * (intptr_t)collArgs.numOutputElements : 0;
  }

  bool UseVerifiableData(CollectiveArgs const& collArgs)
  {
    if (collArgs.funcType != ncclCollAllReduce && collArgs.funcType != ncclCollReduce
        && collArgs.funcType != ncclCollReduceScatter) return false;
    // Custom PreMulSum scalars, bias and constant inputs keep the host reference path
    if (collArgs.options.redOp >= ncclNumOps || collArgs.options.scalarMode >= 0
        || collArgs.options.useBias || collArgs.options.inputConstantValue >= 0) return false;
    return collArgs.dataType == ncclFloat8e4m3 || collArgs.dataType == ncclFloat8e5m2;
  }

  ErrCode VerifiablePrepData(CollectiveArgs& collArgs)
  {
    size_t const numOutputBytes = collArgs.numOutputElements * DataTypeToBytes(collArgs.dataType);
    CHECK_CALL(collArgs.outputGpu.ClearGpuMem(numOutputBytes));
    CHECK_HIP(VerifiableFp8PrepareInput(collArgs.inputGpu.ptr, collArgs.numInputElements,
                                        collArgs.dataType, collArgs.options.redOp,
                                        collArgs.totalRanks, collArgs.globalRank,
                                        VerifiableSeed(collArgs), 0, nullptr));
    CHECK_HIP(hipStreamSynchronize(nullptr));
    collArgs.usesVerifiableData = true;
    return TEST_SUCCESS;
  }

  ErrCode VerifiableValidate(CollectiveArgs& collArgs)
  {
    CHECK_HIP(hipSetDevice(collArgs.deviceId));
    size_t const eltBytes = DataTypeToBytes(collArgs.dataType);
    uint64_t const seed   = VerifiableSeed(collArgs);
    intptr_t const eltIx0 = OutputEltIx0(collArgs);
    ncclRedOp_t const redOp = collArgs.options.redOp;

    // Negative control: a correct collective output must fail once one byte is flipped
    char const* faultEnv = getenv("UT_VERIFIABLE_FAULT");
    if (faultEnv != nullptr && atoi(faultEnv) != 0 && collArgs.numOutputElements > 0)
    {
      uint8_t byte = 0;
      CHECK_HIP(hipMemcpy(&byte, collArgs.outputGpu.ptr, 1, hipMemcpyDeviceToHost));
      byte ^= 0x01;
      CHECK_HIP(hipMemcpy(collArgs.outputGpu.ptr, &byte, 1, hipMemcpyHostToDevice));
      TEST_INFO("[verifiable] FAULT injected into output[0] (rank %d)", collArgs.globalRank);
    }

    int64_t* badEltN = nullptr;
    CHECK_HIP(hipHostMalloc((void**)&badEltN, sizeof(int64_t)));
    *badEltN = 0;
    hipError_t const verifyErr =
      VerifiableFp8Verify(collArgs.outputGpu.ptr, collArgs.numOutputElements, collArgs.dataType,
                          redOp, collArgs.totalRanks, seed, eltIx0, badEltN, nullptr);
    hipError_t const syncErr = hipStreamSynchronize(nullptr);
    int64_t const mismatches = *badEltN;
    CHECK_HIP(hipHostFree(badEltN));
    CHECK_HIP(verifyErr);
    CHECK_HIP(syncErr);
    if (mismatches == 0) return TEST_SUCCESS;

    TEST_ERROR("Mismatch (%ld of %zu elements, seed 0x%lx) for %s", (long)mismatches,
               collArgs.numOutputElements, (unsigned long)seed, collArgs.GetDescription().c_str());

    // Show the first few raw differences; for Avg these may include in-tolerance ones
    size_t const numBytes = collArgs.numOutputElements * eltBytes;
    void* expectedGpu = nullptr;
    CHECK_HIP(hipMalloc(&expectedGpu, numBytes));
    hipError_t const prepErr =
      VerifiableFp8PrepareExpected(expectedGpu, collArgs.numOutputElements, collArgs.dataType,
                                   redOp, collArgs.totalRanks, seed, eltIx0, nullptr);
    std::vector<uint8_t> expected(numBytes), actual(numBytes);
    if (prepErr == hipSuccess && hipStreamSynchronize(nullptr) == hipSuccess
        && hipMemcpy(expected.data(), expectedGpu, numBytes, hipMemcpyDeviceToHost) == hipSuccess
        && hipMemcpy(actual.data(), collArgs.outputGpu.ptr, numBytes, hipMemcpyDeviceToHost) == hipSuccess)
    {
      int shown = 0;
      for (size_t i = 0; i < collArgs.numOutputElements && shown < 4; ++i)
      {
        if (memcmp(&expected[i * eltBytes], &actual[i * eltBytes], eltBytes) == 0) continue;
        uint64_t e = 0, a = 0;
        memcpy(&e, &expected[i * eltBytes], eltBytes);
        memcpy(&a, &actual[i * eltBytes], eltBytes);
        TEST_ERROR("  element %zu: expected bits 0x%lx, got 0x%lx", i, (unsigned long)e, (unsigned long)a);
        ++shown;
      }
    }
    CHECK_HIP(hipFree(expectedGpu));
    return TEST_FAIL;
  }
}
