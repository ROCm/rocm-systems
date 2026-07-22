/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the packed 2-wide fp8/bf8 reduction helpers in
// src/include/rccl_float8.h:
//   hadd2 / hadd2_b          - element-wise sum
//   hminmax2 / hminmax2_b    - element-wise min / max
//   hmul2 / hmul2_b          - element-wise product
//   hpremul2 / hpremul2_b    - element-wise multiply by a broadcast scalar
//
// These operate on fp8x2_storage_t (a uint16_t packing two fp8 bytes: element
// 0 in the low byte, element 1 in the high byte) and, depending on the target,
// take an architecture-specific path (gfx950 packed-f16, gfx942 packed-f32) or
// a scalar per-element fallback. The invariant every path must satisfy is that
// the 2-wide packed result equals doing the same operation one element at a
// time with the pre-existing per-element behavior.
//
// Oracle strategy: the fp8 encoding differs between the host (OCP e4m3/e5m2)
// and the gfx942 device (fnuz), so a host-side reference would not match the
// device result. The oracle is computed *on the device* with the same fp8 type
// as the helper, and it mirrors the *existing* per-element semantics rather
// than an idealized model:
//   - Add:    the scalar hadd/hadd_b - on gfx942 these overflow to NaN via the
//             raw cvt (no downcast clipping), so the packed hadd2 must too.
//   - Min/Max/Mul/PreMul: the saturating rccl_float8/rccl_bfloat8 path (downcast
//             clipping to +/-240 / +/-57344), so the packed helpers must saturate.
// Only decoded float values cross back to the host for comparison, so the test
// is valid on gfx90a (fallback), gfx942 (fnuz) and gfx950 (packed) alike.

#include "DeviceTestBase.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "rccl_float8.h"

namespace RcclUnitTesting
{

enum Fp8Op { OP_ADD = 0, OP_MIN, OP_MAX, OP_MUL, OP_PREMUL };

// ---------------------------------------------------------------------------
// Compile-time selection between the fp8 (e4m3) and bf8 (e5m2) helper family.
// ---------------------------------------------------------------------------
template<bool IsBf8> struct Fp8Traits;

template<> struct Fp8Traits<false> {
  using elem_t = rccl_float8;
  static __device__ fp8x2_storage_t add(fp8x2_storage_t x, fp8x2_storage_t y)             { return hadd2(x, y); }
  static __device__ fp8x2_storage_t minmax(fp8x2_storage_t x, fp8x2_storage_t y, bool mn) { return hminmax2(x, y, mn); }
  static __device__ fp8x2_storage_t mul(fp8x2_storage_t x, fp8x2_storage_t y)             { return hmul2(x, y); }
  static __device__ fp8x2_storage_t premul(fp8x2_storage_t x, float s)                    { return hpremul2(x, s); }
  static __device__ elem_t          scalarAdd(elem_t a, elem_t b)                         { return hadd(a, b); }
};

template<> struct Fp8Traits<true> {
  using elem_t = rccl_bfloat8;
  static __device__ fp8x2_storage_t add(fp8x2_storage_t x, fp8x2_storage_t y)             { return hadd2_b(x, y); }
  static __device__ fp8x2_storage_t minmax(fp8x2_storage_t x, fp8x2_storage_t y, bool mn) { return hminmax2_b(x, y, mn); }
  static __device__ fp8x2_storage_t mul(fp8x2_storage_t x, fp8x2_storage_t y)             { return hmul2_b(x, y); }
  static __device__ fp8x2_storage_t premul(fp8x2_storage_t x, float s)                    { return hpremul2_b(x, s); }
  static __device__ elem_t          scalarAdd(elem_t a, elem_t b)                         { return hadd_b(a, b); }
};

// Decode a packed fp8x2 into its two lanes as floats.
template<bool IsBf8>
__device__ inline void decode2(fp8x2_storage_t v, float& lo, float& hi) {
  union { typename Fp8Traits<IsBf8>::elem_t e[2]; fp8x2_storage_t s; } u;
  u.s = v;
  lo = float(u.e[0]);
  hi = float(u.e[1]);
}

// Round two float lanes back to a packed fp8x2 through the fp8 element type.
template<bool IsBf8>
__device__ inline fp8x2_storage_t encode2(float lo, float hi) {
  using elem_t = typename Fp8Traits<IsBf8>::elem_t;
  union { elem_t e[2]; fp8x2_storage_t s; } u;
  u.e[0] = elem_t(lo);
  u.e[1] = elem_t(hi);
  return u.s;
}

// For each item: compute the packed-helper result and an independent per-lane
// float oracle (rounded back to fp8), then hand both back as decoded floats.
template<bool IsBf8>
__global__ void kFp8Reduce(const fp8x2_storage_t* __restrict__ X,
                           const fp8x2_storage_t* __restrict__ Y,
                           float scalar, int op,
                           float* __restrict__ packedOut,
                           float* __restrict__ refOut, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  fp8x2_storage_t x = X[i];
  fp8x2_storage_t y = (Y != nullptr) ? Y[i] : fp8x2_storage_t(0);

  // ---- packed helper under test ----
  fp8x2_storage_t packed;
  switch (op) {
    case OP_ADD:    packed = Fp8Traits<IsBf8>::add(x, y);         break;
    case OP_MIN:    packed = Fp8Traits<IsBf8>::minmax(x, y, true);  break;
    case OP_MAX:    packed = Fp8Traits<IsBf8>::minmax(x, y, false); break;
    case OP_MUL:    packed = Fp8Traits<IsBf8>::mul(x, y);         break;
    default:        packed = Fp8Traits<IsBf8>::premul(x, scalar); break;
  }
  float p0, p1;
  decode2<IsBf8>(packed, p0, p1);

  // ---- reference: apply the pre-existing per-element behavior ----
  // Add: reduce previously summed with the scalar hadd, whose gfx942 path
  // overflows to NaN (no downcast clipping) - so the reference must too.
  // Min/Max/Mul/PreMul: reduce previously went through the saturating
  // rccl_float8/rccl_bfloat8 path, so the reference saturates.
  fp8x2_storage_t refPacked;
  if (op == OP_ADD) {
    using elem_t = typename Fp8Traits<IsBf8>::elem_t;
    union { elem_t e[2]; fp8x2_storage_t s; } ux, uy, uw;
    ux.s = x;
    uy.s = y;
    uw.e[0] = Fp8Traits<IsBf8>::scalarAdd(ux.e[0], uy.e[0]);
    uw.e[1] = Fp8Traits<IsBf8>::scalarAdd(ux.e[1], uy.e[1]);
    refPacked = uw.s;
  } else {
    float x0, x1, y0, y1;
    decode2<IsBf8>(x, x0, x1);
    decode2<IsBf8>(y, y0, y1);
    float r0, r1;
    switch (op) {
      case OP_MIN:    r0 = fminf(x0, y0);         r1 = fminf(x1, y1);         break;
      case OP_MAX:    r0 = fmaxf(x0, y0);         r1 = fmaxf(x1, y1);         break;
      case OP_MUL:    r0 = x0 * y0;               r1 = x1 * y1;               break;
      default:        r0 = x0 * scalar;           r1 = x1 * scalar;           break;
    }
    refPacked = encode2<IsBf8>(r0, r1);
  }
  float ref0, ref1;
  decode2<IsBf8>(refPacked, ref0, ref1);

  packedOut[2 * i]     = p0;
  packedOut[2 * i + 1] = p1;
  refOut[2 * i]        = ref0;
  refOut[2 * i + 1]    = ref1;
}

class Fp8ReduceTest : public DeviceTestBase {
protected:
  // Compare packed-helper output against the float oracle. NaN==NaN passes.
  static void expectMatch(const std::vector<float>& packed,
                          const std::vector<float>& ref,
                          const char* name, float scalar = 0.0f) {
    int mismatches = 0;
    for (size_t i = 0; i < packed.size(); ++i) {
      float a = packed[i], b = ref[i];
      if (std::isnan(a) && std::isnan(b)) continue;
      if (a != b) {
        if (mismatches < 10)
          ADD_FAILURE() << name << ": lane " << i << " packed=" << a
                        << " oracle=" << b << " scalar=" << scalar;
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << name << ": " << mismatches << " mismatched lanes";
  }

  // Sweep all 256x256 ordered fp8 byte pairs. Lane 0 carries (a,b), lane 1
  // carries the swapped (b,a) so both lanes independently see every pair and a
  // lane-swap bug cannot hide.
  template<bool IsBf8>
  void runTwoInput(int op, const char* name) {
    const int N = 256 * 256;
    std::vector<fp8x2_storage_t> hx(N), hy(N);
    for (int idx = 0; idx < N; ++idx) {
      uint8_t a = static_cast<uint8_t>(idx & 0xFF);
      uint8_t b = static_cast<uint8_t>((idx >> 8) & 0xFF);
      hx[idx] = static_cast<fp8x2_storage_t>(a | (static_cast<uint16_t>(b) << 8));
      hy[idx] = static_cast<fp8x2_storage_t>(b | (static_cast<uint16_t>(a) << 8));
    }
    DeviceBuffer<fp8x2_storage_t> dx(N), dy(N);
    dx.copyFrom(hx);
    dy.copyFrom(hy);
    DeviceBuffer<float> dp(2 * N), dr(2 * N);

    kFp8Reduce<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(dx.ptr, dy.ptr, 0.0f, op,
                                                         dp.ptr, dr.ptr, N);
    syncAndCheck();
    expectMatch(dp.copyTo(), dr.copyTo(), name);
  }

  // Sweep all 256 byte values in both lanes for a set of scalars, including
  // zero, negative, and a magnitude that forces saturation.
  template<bool IsBf8>
  void runPreMul(const char* name) {
    const int N = 256 * 256;
    std::vector<fp8x2_storage_t> hx(N);
    for (int idx = 0; idx < N; ++idx) {
      uint8_t a = static_cast<uint8_t>(idx & 0xFF);
      uint8_t b = static_cast<uint8_t>((idx >> 8) & 0xFF);
      hx[idx] = static_cast<fp8x2_storage_t>(a | (static_cast<uint16_t>(b) << 8));
    }
    DeviceBuffer<fp8x2_storage_t> dx(N);
    dx.copyFrom(hx);
    DeviceBuffer<float> dp(2 * N), dr(2 * N);

    const float scalars[] = {0.0f, 0.5f, 1.0f, 2.0f, -1.0f, -3.5f, 100.0f, 1e30f};
    for (float s : scalars) {
      kFp8Reduce<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(dx.ptr, nullptr, s, OP_PREMUL,
                                                           dp.ptr, dr.ptr, N);
      syncAndCheck();
      expectMatch(dp.copyTo(), dr.copyTo(), name, s);
    }
  }
};

// ---- fp8 (e4m3) ----
TEST_F(Fp8ReduceTest, Fp8Add)    { runTwoInput<false>(OP_ADD, "hadd2"); }
TEST_F(Fp8ReduceTest, Fp8Min)    { runTwoInput<false>(OP_MIN, "hminmax2/min"); }
TEST_F(Fp8ReduceTest, Fp8Max)    { runTwoInput<false>(OP_MAX, "hminmax2/max"); }
TEST_F(Fp8ReduceTest, Fp8Mul)    { runTwoInput<false>(OP_MUL, "hmul2"); }
TEST_F(Fp8ReduceTest, Fp8PreMul) { runPreMul<false>("hpremul2"); }

// ---- bf8 (e5m2) ----
TEST_F(Fp8ReduceTest, Bf8Add)    { runTwoInput<true>(OP_ADD, "hadd2_b"); }
TEST_F(Fp8ReduceTest, Bf8Min)    { runTwoInput<true>(OP_MIN, "hminmax2_b/min"); }
TEST_F(Fp8ReduceTest, Bf8Max)    { runTwoInput<true>(OP_MAX, "hminmax2_b/max"); }
TEST_F(Fp8ReduceTest, Bf8Mul)    { runTwoInput<true>(OP_MUL, "hmul2_b"); }
TEST_F(Fp8ReduceTest, Bf8PreMul) { runPreMul<true>("hpremul2_b"); }

} // namespace RcclUnitTesting
