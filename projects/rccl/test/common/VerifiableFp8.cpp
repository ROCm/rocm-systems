/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "VerifiableFp8.hpp"
#include "rccl_float8.h"
#include <algorithm>
#include <type_traits>

// Design:
// - Each element is hashed from (seed, element index); each rank's value from that hash and
//   its rank, with rank roles rotated per element so every rank's data is exercised.
// - Every value is a small integer times a power of two, bounded so that any partial sum or
//   product over any subset of ranks is exactly representable in FP8 (at most 2^(mantissa+1)-1
//   in integer units, within range). Rounding never occurs, so the result does not depend on
//   accumulation order or intermediate precision, and the check is exact (Avg excepted).
// - Sum: up to 15 (e4m3) / 7 (e5m2) ranks send +-1, one a larger magnitude; others send 0.
//   Prod: one rank sends a random mantissa, up to 3 send +-2, the rest +-1.
//   Min/Max: independent random values per rank. Avg: two ranks send small integers.
// - The expected value is a plain float fold over the ranks' values, which is exact.
// - Kernels take raw FP8 bytes and touch rccl_float8 / rccl_bfloat8 only in device code,
//   where they are the device's native encoding (fnuz or OCP), as RCCL's collectives use
//   (see DeviceDataOps.hpp for why templated kernels on these types cannot be launched).
namespace RcclUnitTesting
{
  namespace
  {
    constexpr int kThreadsPerBlock = 512;

    __device__ uint64_t Mix(uint64_t x)
    {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    template <typename T>
    __device__ uint8_t ToFp8(float x)
    {
      T const v(x);
      return *reinterpret_cast<uint8_t const*>(&v);
    }

    template <typename T>
    __device__ float FromFp8(uint8_t bits)
    {
      return (float)*reinterpret_cast<T const*>(&bits);
    }

    // Rank's value for the element whose hash is h. Values are small integers times powers of
    // two, bounded so that every partial sum or product over any subset of ranks is exactly
    // representable in both FP8 formats: the reduction is exact in any order and precision.
    template <typename T>
    __device__ float Contribution(int redOp, int rankN, int rank, uint64_t h)
    {
      constexpr int M      = std::is_same<T, rccl_bfloat8>::value ? 2 : 3; // Mantissa bits
      constexpr int maxInt = (2 << M) - 1;                                  // Integers exact up to here
      int      const pos  = (rank + (int)(h % rankN)) % rankN; // Position in a per-element rotation
      uint64_t const hr   = Mix(h ^ (uint64_t)(rank + 1));     // Per-rank bits
      float    const sign = (hr & 1) ? -1.0f : 1.0f;
      switch (redOp)
      {
      case ncclSum:
      {
        // Up to maxInt ranks send +-1, the first a larger magnitude, so |partial sums| <= maxInt
        int const numContrib = rankN < maxInt ? rankN : maxInt;
        int const lead = 1 + (int)((h >> 16) % (maxInt - numContrib + 1));
        if (pos >= numContrib) return 0.0f;
        return sign * ldexpf(pos == 0 ? lead : 1, (int)((h >> 24) % 4) - 2);
      }
      case ncclProd:
      {
        // One rank sends a random mantissa, up to 3 others +-2 and the rest +-1
        int const numTwos = (int)((h >> 16) % 4) < rankN - 1 ? (int)((h >> 16) % 4) : rankN - 1;
        if (pos == 0) return sign * ldexpf((1 << M) + (int)((hr >> 8) % (1 << M)), -M - (int)((h >> 24) % 3));
        return sign * (pos <= numTwos ? 2.0f : 1.0f);
      }
      case ncclAvg:
        // Two ranks send small integers; the 1/rankN pre-multiply rounds, hence the tolerance
        return pos < 2 ? (float)(1 + (int)((hr >> 8) % maxInt)) : 0.0f;
      default:
        // Min/Max: random normal values in [2^-4, 16), distinct per rank
        return sign * ldexpf((1 << M) + (int)((hr >> 8) % (1 << M)), (int)((hr >> 16) % 8) - 4 - M);
      }
    }

    template <typename T>
    __device__ uint8_t Expected(int redOp, int rankN, uint64_t h)
    {
      float acc = 0.0f;
      for (int r = 0; r < rankN; ++r)
      {
        float x = Contribution<T>(redOp, rankN, r, h);
        if (redOp == ncclAvg) x = FromFp8<T>(ToFp8<T>(x * (1.0f / rankN)));
        if (r == 0 && redOp != ncclAvg) acc = x;
        else if (redOp == ncclProd)     acc *= x;
        else if (redOp == ncclMax)      acc = fmaxf(acc, x);
        else if (redOp == ncclMin)      acc = fminf(acc, x);
        else                            acc += x;
      }
      return ToFp8<T>(acc);
    }

    __device__ uint64_t ElementHash(uint64_t seed, intptr_t index)
    {
      return Mix(seed ^ Mix((uint64_t)index));
    }

    // rankMe < 0 generates the expected result instead of an input
    template <typename T>
    __device__ void GenerateBody(uint8_t* elts, intptr_t eltN, int redOp, int rankN, int rankMe,
                                 uint64_t seed, intptr_t eltIx0)
    {
      for (intptr_t i = (intptr_t)blockIdx.x * blockDim.x + threadIdx.x; i < eltN; i += (intptr_t)gridDim.x * blockDim.x)
      {
        uint64_t const h = ElementHash(seed, eltIx0 + i);
        elts[i] = rankMe < 0 ? Expected<T>(redOp, rankN, h) : ToFp8<T>(Contribution<T>(redOp, rankN, rankMe, h));
      }
    }

    template <typename T>
    __device__ void VerifyBody(uint8_t const* results, intptr_t eltN, int redOp, int rankN,
                               uint64_t seed, intptr_t eltIx0, int tolerance, int64_t* badEltN)
    {
      unsigned long long bad = 0;
      for (intptr_t i = (intptr_t)blockIdx.x * blockDim.x + threadIdx.x; i < eltN; i += (intptr_t)gridDim.x * blockDim.x)
      {
        int const expected = Expected<T>(redOp, rankN, ElementHash(seed, eltIx0 + i));
        bad += abs((int)results[i] - expected) > tolerance ? 1 : 0;
      }
      atomicAdd((unsigned long long*)badEltN, bad);
    }

    __global__ void GenerateKernel(uint8_t* elts, intptr_t eltN, bool isE5m2, int redOp, int rankN,
                                   int rankMe, uint64_t seed, intptr_t eltIx0)
    {
      if (isE5m2) GenerateBody<rccl_bfloat8>(elts, eltN, redOp, rankN, rankMe, seed, eltIx0);
      else        GenerateBody<rccl_float8> (elts, eltN, redOp, rankN, rankMe, seed, eltIx0);
    }

    __global__ void VerifyKernel(uint8_t const* results, intptr_t eltN, bool isE5m2, int redOp, int rankN,
                                 uint64_t seed, intptr_t eltIx0, int tolerance, int64_t* badEltN)
    {
      if (isE5m2) VerifyBody<rccl_bfloat8>(results, eltN, redOp, rankN, seed, eltIx0, tolerance, badEltN);
      else        VerifyBody<rccl_float8> (results, eltN, redOp, rankN, seed, eltIx0, tolerance, badEltN);
    }

    bool IsSupported(ncclDataType_t dataType, ncclRedOp_t redOp, int rankN)
    {
      return (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2) && rankN > 0 &&
             (redOp == ncclSum || redOp == ncclProd || redOp == ncclMax || redOp == ncclMin || redOp == ncclAvg);
    }

    unsigned NumBlocks(intptr_t eltN)
    {
      return (unsigned)std::min<intptr_t>(1024, (eltN + kThreadsPerBlock - 1) / kThreadsPerBlock);
    }

    hipError_t Generate(void* elts, intptr_t eltN, ncclDataType_t dataType, ncclRedOp_t redOp,
                        int rankN, int rankMe, uint64_t seed, intptr_t eltIx0, hipStream_t stream)
    {
      if (!IsSupported(dataType, redOp, rankN)) return hipErrorInvalidValue;
      if (eltN == 0) return hipSuccess;
      hipLaunchKernelGGL(GenerateKernel, dim3(NumBlocks(eltN)), dim3(kThreadsPerBlock), 0, stream,
                         (uint8_t*)elts, eltN, dataType == ncclFloat8e5m2, (int)redOp, rankN, rankMe,
                         seed, eltIx0);
      return hipGetLastError();
    }
  }

  hipError_t VerifiableFp8PrepareInput(void* elts, intptr_t eltN, ncclDataType_t dataType,
                                       ncclRedOp_t redOp, int rankN, int rankMe,
                                       uint64_t seed, intptr_t eltIx0, hipStream_t stream)
  {
    if (rankMe < 0 || rankMe >= rankN) return hipErrorInvalidValue;
    return Generate(elts, eltN, dataType, redOp, rankN, rankMe, seed, eltIx0, stream);
  }

  hipError_t VerifiableFp8PrepareExpected(void* elts, intptr_t eltN, ncclDataType_t dataType,
                                          ncclRedOp_t redOp, int rankN,
                                          uint64_t seed, intptr_t eltIx0, hipStream_t stream)
  {
    return Generate(elts, eltN, dataType, redOp, rankN, -1, seed, eltIx0, stream);
  }

  hipError_t VerifiableFp8Verify(void const* results, intptr_t eltN, ncclDataType_t dataType,
                                 ncclRedOp_t redOp, int rankN, uint64_t seed, intptr_t eltIx0,
                                 int64_t* badEltN, hipStream_t stream)
  {
    if (!IsSupported(dataType, redOp, rankN)) return hipErrorInvalidValue;
    *badEltN = 0;
    if (eltN == 0) return hipSuccess;
    hipLaunchKernelGGL(VerifyKernel, dim3(NumBlocks(eltN)), dim3(kThreadsPerBlock), 0, stream,
                       (uint8_t const*)results, eltN, dataType == ncclFloat8e5m2, (int)redOp, rankN,
                       seed, eltIx0, redOp == ncclAvg ? 2 : 0, badEltN);
    return hipGetLastError();
  }
}
