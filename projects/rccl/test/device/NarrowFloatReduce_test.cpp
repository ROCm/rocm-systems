/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the packed 2-wide fp8/bf8 reduce helpers in
// src/include/rccl_float8.h:
//   hadd2 / hadd2_b        - sum
//   hminmax2 / hminmax2_b  - min / max
//   hmul2 / hmul2_b        - product
//   hpremul2 / hpremul2_b  - multiply by a broadcast scalar
//
// They operate on fp8x2_storage_t, a uint16_t holding two fp8 bytes (element 0
// in the low byte). Depending on the target they take either the packed
// conversion path (gfx942/gfx950/gfx12xx) or a per-element fallback.
//
// Two invariants are checked.
//
// First, the 2-wide packed result is bit-identical to applying the same
// operation one element at a time via the expressions reduce_kernel.h uses for
// EltPerPack=1. That is what makes the EltPerPack=2 specializations a pure
// vectorization: a reduction result cannot depend on whether a given element
// happened to land in an aligned pair. This is not circular even though both
// widths share rccl_narrow_sat_f32, because the packed path clamps two lanes at
// once and converts with the hardware pack, while the scalar path clamps one
// lane and converts through the HIP fp8 constructor.
//
// Second, saturation is checked directly rather than against a reference: with
// RCCL_NARROW_FP_SATURATE_INF set, no product may come back as Inf, and NaN must
// stay NaN.
//
// The oracle runs on the device, not the host, because the fp8 encoding is not
// portable: gfx942 uses the fnuz variants (e4m3 max 240) while gfx950 and
// gfx12xx use OCP (e4m3 max 448), so a host-computed reference would disagree
// with the device for reasons unrelated to the helpers. Both the helper result
// and the oracle come back as raw bytes and are compared bit-for-bit; decoded
// floats come back too, only to make failures readable and to tell a genuine
// numeric difference apart from a NaN that is merely encoded differently.

#include "DeviceTestBase.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "rccl_float8.h"

namespace RcclUnitTesting
{

enum NarrowOp { OP_ADD = 0, OP_MIN, OP_MAX, OP_MUL, OP_PREMUL };

// Compile-time selection between the fp8 (e4m3) and bf8 (e5m2) helper family.
template<bool IsBf8> struct NarrowTraits;

template<> struct NarrowTraits<false> {
  using elem_t = rccl_float8;
  static __device__ fp8x2_storage_t add(fp8x2_storage_t x, fp8x2_storage_t y)             { return hadd2(x, y); }
  static __device__ fp8x2_storage_t minmax(fp8x2_storage_t x, fp8x2_storage_t y, bool mn) { return hminmax2(x, y, mn); }
  static __device__ fp8x2_storage_t mul(fp8x2_storage_t x, fp8x2_storage_t y)             { return hmul2(x, y); }
  static __device__ fp8x2_storage_t premul(fp8x2_storage_t x, float s)                    { return hpremul2(x, s); }
  static __device__ elem_t          scalarAdd(elem_t a, elem_t b)                         { return hadd(a, b); }
  static constexpr float            kMaxFinite = RCCL_FP8_MAX_FINITE;
};

template<> struct NarrowTraits<true> {
  using elem_t = rccl_bfloat8;
  static __device__ fp8x2_storage_t add(fp8x2_storage_t x, fp8x2_storage_t y)             { return hadd2_b(x, y); }
  static __device__ fp8x2_storage_t minmax(fp8x2_storage_t x, fp8x2_storage_t y, bool mn) { return hminmax2_b(x, y, mn); }
  static __device__ fp8x2_storage_t mul(fp8x2_storage_t x, fp8x2_storage_t y)             { return hmul2_b(x, y); }
  static __device__ fp8x2_storage_t premul(fp8x2_storage_t x, float s)                    { return hpremul2_b(x, s); }
  static __device__ elem_t          scalarAdd(elem_t a, elem_t b)                         { return hadd_b(a, b); }
  static constexpr float            kMaxFinite = RCCL_BF8_MAX_FINITE;
};

template<bool IsBf8>
__device__ inline void decode2(fp8x2_storage_t v, float& lo, float& hi) {
  union { typename NarrowTraits<IsBf8>::elem_t e[2]; fp8x2_storage_t s; } u;
  u.s = v;
  lo = float(u.e[0]);
  hi = float(u.e[1]);
}

// For each pair: run the packed helper, and independently run the per-element
// EltPerPack=1 expression from reduce_kernel.h as the oracle.
// RCCL_FP8_MAX_FINITE differs between the host pass (OCP, 448) and fnuz device
// builds (240), so the device reports the value it actually saturates to.
template<bool IsBf8>
__global__ void kReportMaxFinite(float* out) { *out = NarrowTraits<IsBf8>::kMaxFinite; }

template<bool IsBf8>
__global__ void kNarrowReduce(const fp8x2_storage_t* __restrict__ X,
                              const fp8x2_storage_t* __restrict__ Y,
                              float scalar, int op,
                              fp8x2_storage_t* __restrict__ packedRaw,
                              fp8x2_storage_t* __restrict__ refRaw,
                              float* __restrict__ packedF,
                              float* __restrict__ refF, int n) {
  using elem_t = typename NarrowTraits<IsBf8>::elem_t;
  constexpr float kMax = NarrowTraits<IsBf8>::kMaxFinite;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  fp8x2_storage_t x = X[i];
  fp8x2_storage_t y = (Y != nullptr) ? Y[i] : fp8x2_storage_t(0);

  fp8x2_storage_t packed;
  switch (op) {
    case OP_ADD: packed = NarrowTraits<IsBf8>::add(x, y);          break;
    case OP_MIN: packed = NarrowTraits<IsBf8>::minmax(x, y, true);  break;
    case OP_MAX: packed = NarrowTraits<IsBf8>::minmax(x, y, false); break;
    case OP_MUL: packed = NarrowTraits<IsBf8>::mul(x, y);          break;
    default:     packed = NarrowTraits<IsBf8>::premul(x, scalar);   break;
  }

  union { elem_t e[2]; fp8x2_storage_t s; } ux, uy, uw;
  ux.s = x;
  uy.s = y;
  for (int l = 0; l < 2; ++l) {
    float a = float(ux.e[l]);
    float b = float(uy.e[l]);
    switch (op) {
      // Sum's EltPerPack=1 form is the scalar hadd helper, not a float add, so
      // the oracle uses hadd too.
      case OP_ADD: uw.e[l] = NarrowTraits<IsBf8>::scalarAdd(ux.e[l], uy.e[l]); break;
      // Min/Max return one of their inputs, so they need no clamp; Prod and
      // PreMulSum go through the same saturation the 1-wide path now applies.
      case OP_MIN: uw.e[l] = elem_t(fminf(a, b));                              break;
      case OP_MAX: uw.e[l] = elem_t(fmaxf(a, b));                              break;
      case OP_MUL: uw.e[l] = elem_t(rccl_narrow_sat_f32(a * b, kMax));         break;
      default:     uw.e[l] = elem_t(rccl_narrow_sat_f32(a * scalar, kMax));    break;
    }
  }

  packedRaw[i] = packed;
  refRaw[i]    = uw.s;
  decode2<IsBf8>(packed, packedF[2 * i], packedF[2 * i + 1]);
  decode2<IsBf8>(uw.s, refF[2 * i], refF[2 * i + 1]);
}

class NarrowFloatReduceTest : public DeviceTestBase {
protected:
  template<bool IsBf8>
  float deviceMaxFinite() {
    DeviceBuffer<float> d(1);
    kReportMaxFinite<IsBf8><<<1, 1>>>(d.ptr);
    syncAndCheck();
    return d.download();
  }

  // Bit-for-bit comparison. A mismatch where both lanes decode to NaN is
  // reported separately: it is a difference in NaN encoding, not in value.
  static void expectBitMatch(const std::vector<fp8x2_storage_t>& packedRaw,
                             const std::vector<fp8x2_storage_t>& refRaw,
                             const std::vector<float>& packedF,
                             const std::vector<float>& refF,
                             const char* name, float scalar, bool haveScalar) {
    int valueMismatches = 0;
    int nanEncodingOnly = 0;
    for (size_t i = 0; i < packedRaw.size(); ++i) {
      if (packedRaw[i] == refRaw[i]) continue;
      for (int l = 0; l < 2; ++l) {
        float p = packedF[2 * i + l];
        float r = refF[2 * i + l];
        uint8_t pb = static_cast<uint8_t>((packedRaw[i] >> (8 * l)) & 0xFF);
        uint8_t rb = static_cast<uint8_t>((refRaw[i] >> (8 * l)) & 0xFF);
        if (pb == rb) continue;
        if (std::isnan(p) && std::isnan(r)) {
          ++nanEncodingOnly;
          continue;
        }
        if (valueMismatches < 10) {
          std::string ctx;
          if (haveScalar) ctx = " scalar=" + std::to_string(scalar);
          ADD_FAILURE() << name << ": pair " << i << " lane " << l << " packed=0x" << std::hex
                        << unsigned(pb) << " (" << std::dec << p << ") oracle=0x" << std::hex
                        << unsigned(rb) << " (" << std::dec << r << ")" << ctx;
        }
        ++valueMismatches;
      }
    }
    EXPECT_EQ(valueMismatches, 0) << name << ": " << valueMismatches << " lanes differ in value";
    if (nanEncodingOnly != 0)
      GTEST_LOG_(INFO) << name << ": " << nanEncodingOnly
                       << " lanes are NaN in both but with different encodings";
  }

  // Saturation checked on its own terms rather than against a reference: with
  // RCCL_NARROW_FP_SATURATE_INF nothing may come back as Inf, and every finite
  // result must be within range. NaN is allowed through unchanged.
  static void expectSaturated(const std::vector<float>& out, float maxFinite, const char* name) {
    int infs = 0, oor = 0;
    for (float v : out) {
      if (std::isnan(v)) continue;
      if (std::isinf(v)) ++infs;
      else if (std::fabs(v) > maxFinite) ++oor;
    }
#if RCCL_NARROW_FP_SATURATE_INF
    EXPECT_EQ(infs, 0) << name << ": " << infs << " results are Inf despite saturation being on";
#endif
    EXPECT_EQ(oor, 0) << name << ": " << oor << " finite results exceed " << maxFinite;
  }

  // Every ordered pair of fp8 bytes. Lane 0 carries (a,b) and lane 1 the
  // swapped (b,a), so each lane independently sees all 65536 pairs and a
  // lane-swap bug cannot cancel out.
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
    DeviceBuffer<fp8x2_storage_t> dx(N), dy(N), dpr(N), drr(N);
    dx.copyFrom(hx);
    dy.copyFrom(hy);
    DeviceBuffer<float> dpf(2 * N), drf(2 * N);

    kNarrowReduce<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(
        dx.ptr, dy.ptr, 0.0f, op, dpr.ptr, drr.ptr, dpf.ptr, drf.ptr, N);
    syncAndCheck();
    std::vector<float> packedF = dpf.copyTo();
    expectBitMatch(dpr.copyTo(), drr.copyTo(), packedF, drf.copyTo(), name, 0.0f, false);
    if (op == OP_MUL) expectSaturated(packedF, deviceMaxFinite<IsBf8>(), name);
  }

  // Every fp8 byte in both lanes, against scalars that cover zero, negative,
  // fractional, and magnitudes large enough to force saturation.
  template<bool IsBf8>
  void runPreMul(const char* name) {
    const int N = 256 * 256;
    std::vector<fp8x2_storage_t> hx(N);
    for (int idx = 0; idx < N; ++idx) {
      uint8_t a = static_cast<uint8_t>(idx & 0xFF);
      uint8_t b = static_cast<uint8_t>((idx >> 8) & 0xFF);
      hx[idx] = static_cast<fp8x2_storage_t>(a | (static_cast<uint16_t>(b) << 8));
    }
    DeviceBuffer<fp8x2_storage_t> dx(N), dpr(N), drr(N);
    dx.copyFrom(hx);
    DeviceBuffer<float> dpf(2 * N), drf(2 * N);

    const float maxFinite = deviceMaxFinite<IsBf8>();
    const float scalars[] = {0.0f, -0.0f, 0.5f, 1.0f, 2.0f, -1.0f, -3.5f, 100.0f, 1e30f, -1e30f};
    for (float s : scalars) {
      kNarrowReduce<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(
          dx.ptr, nullptr, s, OP_PREMUL, dpr.ptr, drr.ptr, dpf.ptr, drf.ptr, N);
      syncAndCheck();
      std::vector<float> packedF = dpf.copyTo();
      expectBitMatch(dpr.copyTo(), drr.copyTo(), packedF, drf.copyTo(), name, s, true);
      expectSaturated(packedF, maxFinite, name);
    }
  }
};

// ---- fp8 (e4m3) ----
TEST_F(NarrowFloatReduceTest, Fp8Add)    { runTwoInput<false>(OP_ADD, "hadd2"); }
TEST_F(NarrowFloatReduceTest, Fp8Min)    { runTwoInput<false>(OP_MIN, "hminmax2/min"); }
TEST_F(NarrowFloatReduceTest, Fp8Max)    { runTwoInput<false>(OP_MAX, "hminmax2/max"); }
TEST_F(NarrowFloatReduceTest, Fp8Mul)    { runTwoInput<false>(OP_MUL, "hmul2"); }
TEST_F(NarrowFloatReduceTest, Fp8PreMul) { runPreMul<false>("hpremul2"); }

// ---- bf8 (e5m2) ----
TEST_F(NarrowFloatReduceTest, Bf8Add)    { runTwoInput<true>(OP_ADD, "hadd2_b"); }
TEST_F(NarrowFloatReduceTest, Bf8Min)    { runTwoInput<true>(OP_MIN, "hminmax2_b/min"); }
TEST_F(NarrowFloatReduceTest, Bf8Max)    { runTwoInput<true>(OP_MAX, "hminmax2_b/max"); }
TEST_F(NarrowFloatReduceTest, Bf8Mul)    { runTwoInput<true>(OP_MUL, "hmul2_b"); }
TEST_F(NarrowFloatReduceTest, Bf8PreMul) { runPreMul<true>("hpremul2_b"); }

} // namespace RcclUnitTesting
