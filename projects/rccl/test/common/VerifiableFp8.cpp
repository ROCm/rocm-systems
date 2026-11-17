/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "VerifiableFp8.hpp"
#include "DeviceDataOps.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

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
// - Kernels are templated on Fp8Bits' format flag, not on rccl_float8 / rccl_bfloat8 (see
//   DeviceDataOps.hpp), and convert only in device code, where the native type is the
//   device's encoding (fnuz or OCP), as RCCL's collectives use.
// - When this file's kernels cannot run on the current device, or UT_DEVICE_DATA=0, the same
//   per-element functions run on the host, converting with the device's encoding explicitly.
namespace RcclUnitTesting
{
  namespace
  {
    constexpr int kThreadsPerBlock = 512;

    // FP8 conversion in the device's native encoding (device code only)
    template <bool IsE5m2>
    struct DeviceFp8Codec
    {
      __device__ uint8_t Encode(float x) const
      {
        using B = Fp8Bits<IsE5m2>;
        return B::From(typename B::Native(x)).bits;
      }
      __device__ float Decode(uint8_t bits) const { return (float)Fp8Bits<IsE5m2>{bits}.Get(); }
    };

    // Host conversion in the encoding the device uses
    struct HostFp8Codec
    {
      bool isE5m2;
      bool fnuz;
      uint8_t Encode(float x) const { return rcclFloatToFp8(x, isE5m2, fnuz); }
      float Decode(uint8_t bits) const { return rcclFp8ToFloat(bits, isE5m2, fnuz); }
    };

    __host__ __device__ uint64_t Mix(uint64_t x)
    {
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    // Rank's value for the element whose hash is h. Values are small integers times powers of
    // two, bounded so that every partial sum or product over any subset of ranks is exactly
    // representable in both FP8 formats: the reduction is exact in any order and precision.
    __host__ __device__ float Contribution(bool isE5m2, int redOp, int rankN, int rank, uint64_t h)
    {
      int      const M      = isE5m2 ? 2 : 3;                    // Mantissa bits
      int      const maxInt = (2 << M) - 1;                      // Integers exact up to here
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

    template <typename Codec>
    __host__ __device__ uint8_t Expected(Codec const& codec, bool isE5m2, int redOp, int rankN, uint64_t h)
    {
      float acc = 0.0f;
      for (int r = 0; r < rankN; ++r)
      {
        float x = Contribution(isE5m2, redOp, rankN, r, h);
        if (redOp == ncclAvg) x = codec.Decode(codec.Encode(x * (1.0f / rankN)));
        if (r == 0 && redOp != ncclAvg) acc = x;
        else if (redOp == ncclProd)     acc *= x;
        else if (redOp == ncclMax)      acc = fmaxf(acc, x);
        else if (redOp == ncclMin)      acc = fminf(acc, x);
        else                            acc += x;
      }
      return codec.Encode(acc);
    }

    __host__ __device__ uint64_t ElementHash(uint64_t seed, intptr_t index)
    {
      return Mix(seed ^ Mix((uint64_t)index));
    }

    // rankMe < 0 generates the expected result instead of an input
    template <typename Codec>
    __host__ __device__ uint8_t GenerateElt(Codec const& codec, bool isE5m2, int redOp, int rankN, int rankMe,
                                            uint64_t seed, intptr_t eltIx)
    {
      uint64_t const h = ElementHash(seed, eltIx);
      return rankMe < 0 ? Expected(codec, isE5m2, redOp, rankN, h)
                        : codec.Encode(Contribution(isE5m2, redOp, rankN, rankMe, h));
    }

    template <typename Codec>
    __host__ __device__ bool IsBadElt(Codec const& codec, bool isE5m2, uint8_t result, int redOp, int rankN,
                                      uint64_t seed, intptr_t eltIx, int tolerance)
    {
      int const expected = Expected(codec, isE5m2, redOp, rankN, ElementHash(seed, eltIx));
      return abs((int)result - expected) > tolerance;
    }

    template <bool IsE5m2>
    __global__ void GenerateKernel(uint8_t* elts, intptr_t eltN, int redOp, int rankN,
                                   int rankMe, uint64_t seed, intptr_t eltIx0)
    {
      DeviceFp8Codec<IsE5m2> const codec;
      for (intptr_t i = (intptr_t)blockIdx.x * blockDim.x + threadIdx.x; i < eltN; i += (intptr_t)gridDim.x * blockDim.x)
      {
        elts[i] = GenerateElt(codec, IsE5m2, redOp, rankN, rankMe, seed, eltIx0 + i);
      }
    }

    template <bool IsE5m2>
    __global__ void VerifyKernel(uint8_t const* results, intptr_t eltN, int redOp, int rankN,
                                 uint64_t seed, intptr_t eltIx0, int tolerance, int64_t* badEltN)
    {
      DeviceFp8Codec<IsE5m2> const codec;
      unsigned long long bad = 0;
      for (intptr_t i = (intptr_t)blockIdx.x * blockDim.x + threadIdx.x; i < eltN; i += (intptr_t)gridDim.x * blockDim.x)
      {
        bad += IsBadElt(codec, IsE5m2, results[i], redOp, rankN, seed, eltIx0 + i, tolerance) ? 1 : 0;
      }
      atomicAdd((unsigned long long*)badEltN, bad);
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

    // Same answer as UtDeviceDataEnabled() for this file's code object. Not shared with it
    // because rccl-UnitTestsMPI builds this file without PtrUnion.cpp.
    bool UseDeviceKernels()
    {
      static KernelLoadCache kernels;
      return UtDeviceDataRequested() &&
             kernels.Loadable(reinterpret_cast<void const*>(&GenerateKernel<false>), "Verifiable FP8");
    }

    // Host mirror of rcclFp8DeviceIsFnuz (archinfo.h), whose IsArchMatch librccl does not export
    bool CurrentDeviceIsFnuz()
    {
      int dev = 0;
      hipDeviceProp_t prop = {};
      if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess)
      {
        (void)hipGetLastError();
        return true;
      }
      for (char const* ocpArch : {"gfx950", "gfx1200", "gfx1201", "gfx1250"})
      {
        if (strncmp(prop.gcnArchName, ocpArch, strlen(ocpArch)) == 0) return false;
      }
      return true;
    }

    hipError_t Generate(void* elts, intptr_t eltN, ncclDataType_t dataType, ncclRedOp_t redOp,
                        int rankN, int rankMe, uint64_t seed, intptr_t eltIx0, hipStream_t stream)
    {
      if (!IsSupported(dataType, redOp, rankN)) return hipErrorInvalidValue;
      if (eltN == 0) return hipSuccess;
      bool const isE5m2 = (dataType == ncclFloat8e5m2);
      if (UseDeviceKernels())
      {
        auto const kernel = isE5m2 ? GenerateKernel<true> : GenerateKernel<false>;
        hipLaunchKernelGGL(kernel, dim3(NumBlocks(eltN)), dim3(kThreadsPerBlock), 0, stream,
                           (uint8_t*)elts, eltN, (int)redOp, rankN, rankMe, seed, eltIx0);
        return hipGetLastError();
      }

      HostFp8Codec const codec{isE5m2, CurrentDeviceIsFnuz()};
      std::vector<uint8_t> host(eltN);
      for (intptr_t i = 0; i < eltN; ++i)
      {
        host[i] = GenerateElt(codec, isE5m2, (int)redOp, rankN, rankMe, seed, eltIx0 + i);
      }
      hipError_t err = hipMemcpyAsync(elts, host.data(), eltN, hipMemcpyHostToDevice, stream);
      if (err == hipSuccess) err = hipStreamSynchronize(stream);
      return err;
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
    bool const isE5m2    = (dataType == ncclFloat8e5m2);
    int  const tolerance = redOp == ncclAvg ? 2 : 0;
    if (UseDeviceKernels())
    {
      auto const kernel = isE5m2 ? VerifyKernel<true> : VerifyKernel<false>;
      hipLaunchKernelGGL(kernel, dim3(NumBlocks(eltN)), dim3(kThreadsPerBlock), 0, stream,
                         (uint8_t const*)results, eltN, (int)redOp, rankN,
                         seed, eltIx0, tolerance, badEltN);
      return hipGetLastError();
    }

    std::vector<uint8_t> host(eltN);
    hipError_t err = hipMemcpyAsync(host.data(), results, eltN, hipMemcpyDeviceToHost, stream);
    if (err == hipSuccess) err = hipStreamSynchronize(stream);
    if (err != hipSuccess) return err;
    HostFp8Codec const codec{isE5m2, CurrentDeviceIsFnuz()};
    int64_t bad = 0;
    for (intptr_t i = 0; i < eltN; ++i)
    {
      bad += IsBadElt(codec, isE5m2, host[i], (int)redOp, rankN, seed, eltIx0 + i, tolerance) ? 1 : 0;
    }
    *badEltN = bad;
    return hipSuccess;
  }
}
