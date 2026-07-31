/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exhaustive equivalence test for the 2-wide 16-bit reduce specializations in
// src/device/reduce_kernel.h.
//
// A 16-bit element takes 65536 values, so a binary op on a pair of them has
// 2^32 = 4294967296 possible inputs, and all of them are swept here. There is
// nothing left to sample: the claim these specializations make is that a
// reduction result does not depend on whether an element happened to land in an
// aligned pair, and after this test that claim holds for every input the
// hardware can present, not for a chosen subset of them.
//
// The oracle is Apply_Reduce<Fn, 1> on each element separately. That is exactly
// what the dispatcher did before an EltPerPack=2 specialization existed -- the
// generic template recurses into the 1-wide case -- so agreement with it is also
// the statement that the answers have not moved.
//
// Both lanes are checked independently: lane 0 is given (a, b) and lane 1 the
// swapped (b, a), so each lane position sees all 2^32 ordered pairs over the
// sweep and a helper that transposes, duplicates or drops a lane cannot hide.
//
// Comparison is on raw bytes. Decoded floats would lose the distinction between
// -0 and +0 and between NaN payloads, and those are where narrow-float
// conversion differences live: a packed convert that quiets a signaling NaN
// where the scalar one passes it through would go unnoticed.
//
// Two classes of difference are allowed, and only these two. Both are cases
// where the operands carry no information that a reduction can preserve in the
// first place, and both are counted and printed so that a change in how much of
// the input space falls into them is visible rather than silently absorbed.
//
// The first is the payload when both operands are NaN. Which payload survives
// depends on the order the operands reach the instruction, and for a commutative
// operation that order is the compiler's to choose -- it hands the two 1-wide
// calls their operands in the same order and they return the same payload, while
// the packed lanes keep the order they were written in and return one payload
// each. For f16 min and max the 1-wide path does not return a payload at all:
// __hmin answers two NaNs with a canonical NaN, where the hardware's packed min
// returns one of the two. IEEE 754 leaves the choice unspecified.
//
// The second is the sign of a zero out of min and max when the operands are +0
// and -0. __hmin returns whichever operand came first, so the answer already
// depends on the order the reduction happened to visit the ranks in; the packed
// instruction returns -0 for min and +0 for max whatever the order. Only two of
// the 2^32 inputs are in this class.
//
// Everything else must match exactly, including the payload when exactly one
// operand is NaN and the sign of a zero out of sum and product.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

// Generated header: defines RCCL_BFLOAT16, which gates the bf16 specializations
// in reduce_kernel.h. Without it the generic template is instantiated instead
// and the sweep would measure the wrong code.
#include "nccl.h"

// reduce_kernel.h refers to hip_bfloat16, which src/include/device.h aliases to
// __hip_bfloat16 on ROCm 6 and newer. That alias is replicated here rather than
// including device.h, which would pull in the library's entire header chain.
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

constexpr uint64_t kTotalPairs = 1ull << 32;
constexpr uint64_t kNoIndex = ~0ull;

// Exponent all ones with a nonzero significand, in whichever field layout the
// type uses.
template <typename T> struct Narrow16;
template <> struct Narrow16<hip_bfloat16> {
  static constexpr uint16_t kExp = 0x7F80, kSig = 0x007F;
};
template <> struct Narrow16<half> {
  static constexpr uint16_t kExp = 0x7C00, kSig = 0x03FF;
};

template <typename T>
__device__ __forceinline__ bool isNaN16(uint16_t bits) {
  return (bits & Narrow16<T>::kExp) == Narrow16<T>::kExp && (bits & Narrow16<T>::kSig) != 0;
}

__device__ __forceinline__ bool isZero16(uint16_t bits) { return (bits & 0x7FFFu) == 0; }

// One pair per iteration, grid-strided so the launch geometry is independent of
// the sweep size. Differing lanes are counted rather than collected, split into
// the two allowed classes and everything else, with the lowest offending index
// kept so a failure names a reproducible input.
template <typename Fn>
__global__ void kSweep16(uint64_t total, uint64_t opArg, unsigned long long* mismatches,
                         unsigned long long* nanPayload, unsigned long long* zeroSign,
                         unsigned long long* firstBad) {
  using T = typename Fn::EltType;
  Fn fn(opArg);
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    const uint32_t a = uint32_t(i & 0xFFFFu);
    const uint32_t b = uint32_t((i >> 16) & 0xFFFFu);

    BytePack<4> ap, bp;
    ap.native = a | (b << 16);
    bp.native = b | (a << 16);
    const uint32_t got = Apply_Reduce<Fn, 2>::reduce(fn, ap, bp).native;

    BytePack<2> lx, ly, hx, hy;
    lx.native = uint16_t(a);
    ly.native = uint16_t(b);
    hx.native = uint16_t(b);
    hy.native = uint16_t(a);
    const uint32_t want = uint32_t(Apply_Reduce<Fn, 1>::reduce(fn, lx, ly).native) |
                          (uint32_t(Apply_Reduce<Fn, 1>::reduce(fn, hx, hy).native) << 16);

    if (got == want) continue;

    // The lanes hold the same two operands in opposite order, so classifying the
    // operands once covers both; what can differ per lane is only which operand
    // the result came from.
    const bool bothNaN = isNaN16<T>(uint16_t(a)) && isNaN16<T>(uint16_t(b));
    const bool bothZero = isZero16(uint16_t(a)) && isZero16(uint16_t(b));
    for (int lane = 0; lane < 2; ++lane) {
      const uint16_t g = uint16_t((got >> (16 * lane)) & 0xFFFFu);
      if (g == uint16_t((want >> (16 * lane)) & 0xFFFFu)) continue;
      if (bothNaN && isNaN16<T>(g)) {
        atomicAdd(nanPayload, 1ull);
      } else if (bothZero && isZero16(g)) {
        atomicAdd(zeroSign, 1ull);
      } else {
        atomicAdd(mismatches, 1ull);
        atomicMin(firstBad, (unsigned long long)i);
      }
    }
  }
}

// Re-runs a single pair so a failure can report what each width produced.
template <typename Fn>
__global__ void kOnePair(uint64_t i, uint64_t opArg, uint32_t* got, uint32_t* want) {
  Fn fn(opArg);
  const uint32_t a = uint32_t(i & 0xFFFFu);
  const uint32_t b = uint32_t((i >> 16) & 0xFFFFu);
  BytePack<4> ap, bp;
  ap.native = a | (b << 16);
  bp.native = b | (a << 16);
  *got = Apply_Reduce<Fn, 2>::reduce(fn, ap, bp).native;
  BytePack<2> lx, ly, hx, hy;
  lx.native = uint16_t(a);
  ly.native = uint16_t(b);
  hx.native = uint16_t(b);
  hy.native = uint16_t(a);
  *want = uint32_t(Apply_Reduce<Fn, 1>::reduce(fn, lx, ly).native) |
          (uint32_t(Apply_Reduce<Fn, 1>::reduce(fn, hx, hy).native) << 16);
}

// Host-side decode, only used to make failures readable.
template <typename T> float toFloat(uint16_t bits);

template <>
float toFloat<hip_bfloat16>(uint16_t bits) {
  // A bf16 is the top half of the f32 with the same value.
  union {
    uint32_t u;
    float f;
  } u;
  u.u = uint32_t(bits) << 16;
  return u.f;
}

template <>
float toFloat<half>(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return float(*reinterpret_cast<__half*>(&raw));
}

template <typename T>
std::string describe(uint64_t i) {
  const uint16_t a = uint16_t(i & 0xFFFFu);
  const uint16_t b = uint16_t((i >> 16) & 0xFFFFu);
  char buf[128];
  std::snprintf(buf, sizeof(buf), "a=0x%04x (%g) b=0x%04x (%g)", a, toFloat<T>(a), b,
                toFloat<T>(b));
  return buf;
}

} // namespace

class SixteenBitReduceTest : public DeviceTestBase {
protected:
  // Enough threads to fill any current device several times over; each one walks
  // its share of the 2^32 pairs.
  static constexpr int kBlocks = 8192;

  template <typename Fn>
  void sweep(uint64_t opArg, const char* name) {
    DeviceBuffer<unsigned long long> mismatches(1), nanPayload(1), zeroSign(1), firstBad(1);
    mismatches.upload(0ull);
    nanPayload.upload(0ull);
    zeroSign.upload(0ull);
    firstBad.upload(kNoIndex);

    kSweep16<Fn><<<kBlocks, kDefaultBlockSize>>>(kTotalPairs, opArg, mismatches.ptr,
                                                 nanPayload.ptr, zeroSign.ptr, firstBad.ptr);
    syncAndCheck();

    const unsigned long long nans = nanPayload.download(), zeros = zeroSign.download();
    if (nans) std::printf("%s: %llu lanes of two NaN operands differ in payload only\n", name, nans);
    if (zeros) std::printf("%s: %llu lanes of +0 and -0 differ in the sign of the zero\n", name, zeros);

    const unsigned long long bad = mismatches.download();
    if (bad == 0) return;

    const unsigned long long at = firstBad.download();
    DeviceBuffer<uint32_t> got(1), want(1);
    kOnePair<Fn><<<1, 1>>>(at, opArg, got.ptr, want.ptr);
    syncAndCheck();
    ADD_FAILURE() << name << ": 2-wide disagrees with 1-wide on " << bad << " lanes across "
                  << kTotalPairs << " input pairs; lowest is " << describe<typename Fn::EltType>(at)
                  << ", 2-wide gave 0x" << std::hex << got.download()
                  << " where two 1-wide calls give 0x" << want.download() << std::dec
                  << " (lanes packed high:low)";
  }
};

// opArg reaches FuncMinMax as its selector: even means min, odd means max.
TEST_F(SixteenBitReduceTest, Bf16Sum) { sweep<FuncSum<hip_bfloat16>>(0, "bf16 sum"); }
TEST_F(SixteenBitReduceTest, Bf16Prod) { sweep<FuncProd<hip_bfloat16>>(0, "bf16 prod"); }
TEST_F(SixteenBitReduceTest, Bf16Min) { sweep<FuncMinMax<hip_bfloat16>>(0, "bf16 min"); }
TEST_F(SixteenBitReduceTest, Bf16Max) { sweep<FuncMinMax<hip_bfloat16>>(1, "bf16 max"); }

TEST_F(SixteenBitReduceTest, F16Sum) { sweep<FuncSum<half>>(0, "f16 sum"); }
TEST_F(SixteenBitReduceTest, F16Prod) { sweep<FuncProd<half>>(0, "f16 prod"); }
TEST_F(SixteenBitReduceTest, F16Min) { sweep<FuncMinMax<half>>(0, "f16 min"); }
TEST_F(SixteenBitReduceTest, F16Max) { sweep<FuncMinMax<half>>(1, "f16 max"); }

} // namespace RcclUnitTesting
