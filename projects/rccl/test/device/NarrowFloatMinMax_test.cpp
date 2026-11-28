/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exhaustive test of fp8/bf8 min and max in src/device/reduce_kernel.h.
//
// The oracle is the expression the 1-wide specialization used to be: widen
// both operands to f32, fminf/fmaxf, narrow. Every one of the 65536 ordered
// byte pairs is run through Apply_Reduce at widths 1 and 4 (width 4 reaches
// any 2-wide specialization), with the operands swapped in alternate lanes,
// and compared on raw bytes.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "nccl.h"

#if !defined(_HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_) && !defined(_HIP_BFLOAT16_H_)
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_
#define _HIP_BFLOAT16_H_
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 hip_bfloat16;
#endif

#include "reduce_kernel.h"

namespace RcclUnitTesting
{

namespace {

constexpr uint32_t kPairs = 1u << 16;

template <typename T>
__device__ uint8_t oracle(uint8_t a, uint8_t b, bool isMin) {
  T x, y;
  reinterpret_cast<uint8_t&>(x) = a;
  reinterpret_cast<uint8_t&>(y) = b;
  T r(isMin ? fminf(float(x), float(y)) : fmaxf(float(x), float(y)));
  return reinterpret_cast<const uint8_t&>(r);
}

// Templated on a bool, not the type: host code sees the OCP typedefs, so a
// type parameter would mangle differently on the host and on fnuz devices.
template <bool Bf8>
__global__ void kSweep(uint64_t opArg, unsigned long long* bad, uint32_t* firstBad) {
  using T = typename std::conditional<Bf8, rccl_bfloat8, rccl_float8>::type;
  FuncMinMax<T> fn(opArg);
  const bool isMin = (opArg & 1) == 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kPairs; i += gridDim.x * blockDim.x) {
    const uint8_t a = i & 0xff, b = i >> 8;
    const uint8_t want = oracle<T>(a, b, isMin), wantSwapped = oracle<T>(b, a, isMin);

    BytePack<1> x1, y1;
    x1.native = a;
    y1.native = b;
    const uint8_t got1 = Apply_Reduce<FuncMinMax<T>, 1>::reduce(fn, x1, y1).native;

    BytePack<4> x4, y4;
    x4.native = a | (b << 8) | (a << 16) | (b << 24);
    y4.native = b | (a << 8) | (b << 16) | (a << 24);
    const uint32_t got4 = Apply_Reduce<FuncMinMax<T>, 4>::reduce(fn, x4, y4).native;
    const uint32_t want4 = want | (wantSwapped << 8) | (want << 16) | (wantSwapped << 24);

    if (got1 != want || got4 != want4) {
      atomicAdd(bad, 1ull);
      atomicMin(firstBad, i);
    }
  }
}

} // namespace

class NarrowFloatMinMaxTest : public DeviceTestBase {
protected:
  template <bool Bf8>
  void sweep(uint64_t opArg, const char* name) {
    DeviceBuffer<unsigned long long> bad(1);
    DeviceBuffer<uint32_t> firstBad(1);
    bad.upload(0ull);
    firstBad.upload(~0u);
    kSweep<Bf8><<<256, kDefaultBlockSize>>>(opArg, bad.ptr, firstBad.ptr);
    syncAndCheck();
    const unsigned long long n = bad.download();
    if (n == 0) return;
    const uint32_t at = firstBad.download();
    ADD_FAILURE() << name << ": " << n << " of " << kPairs << " pairs differ from fminf/fmaxf; lowest is a=0x"
                  << std::hex << (at & 0xff) << " b=0x" << (at >> 8) << std::dec;
  }
};

// opArg reaches FuncMinMax as its selector: even means min, odd means max.
TEST_F(NarrowFloatMinMaxTest, Fp8Min) { sweep<false>(0, "fp8 min"); }
TEST_F(NarrowFloatMinMaxTest, Fp8Max) { sweep<false>(1, "fp8 max"); }
TEST_F(NarrowFloatMinMaxTest, Bf8Min) { sweep<true>(0, "bf8 min"); }
TEST_F(NarrowFloatMinMaxTest, Bf8Max) { sweep<true>(1, "bf8 max"); }

} // namespace RcclUnitTesting
