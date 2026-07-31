/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression tests for the fp8/bf8 Sum helpers in src/include/rccl_float8.h:
//   hadd / hadd_b    - one element
//   hadd2 / hadd2_b  - two elements packed into a uint16_t, element 0 low
//
// These are the helpers FuncSum dispatches to. Their packed conversions round
// rather than saturate, so a sum of two finite operands that leaves the
// destination's range used to come back as Inf, or as NaN for e4m3, which has no
// Inf encoding: Sum(448, 18) returned 0x7f, a NaN, instead of the 0x7e that
// encodes 448. All 65536 ordered byte pairs are swept, which covers every input
// the helpers can be given.
//
// Three independent things are checked.
//
// First, the result matches what elem_t(float(a) + float(b)) returns, that is,
// an f32 add narrowed by the HIP fp8 constructor. That constructor saturates in
// __HIP_SATFINITE mode, and it is the path Prod, MinMax and PreMulSum already
// take, so it is both the behavior Sum is meant to share and an oracle that
// shares no code with the helpers under test.
//
// Second, the 1-wide and 2-wide results agree byte for byte, so a reduction's
// answer cannot depend on whether an element happened to land in an aligned
// pair. On gfx942 it did: the 1-wide helper recovered the packed bytes through
// rccl_float8's numeric int constructor, which reinterpreted them as a value.
//
// Third, the promised semantics, without reference to any implementation, since
// an oracle can only show that two paths agree and not that either is right:
// finite operands give a finite in-range result, saturated exactly to
// +/-max_finite when the exact sum is out of range; an Inf operand gives Inf,
// except where the destination cannot hold one; a NaN operand gives NaN.
//
// The oracle runs on the device because the encoding is not portable: gfx942
// uses the fnuz variants, where e4m3 stops at 240 and there are no Inf
// encodings at all, while gfx950 and gfx12xx use OCP, where it stops at 448. A
// host-computed reference would disagree for reasons having nothing to do with
// the helpers. The operands and results therefore come back both as raw bytes,
// for exact comparison, and as floats the device decoded, so the host can apply
// the rules without knowing which encoding is in force.

#include "DeviceTestBase.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "rccl_float8.h"

namespace RcclUnitTesting
{

// Compile-time selection between the fp8 (e4m3) and bf8 (e5m2) helper family.
template<bool IsBf8> struct SumTraits;

template<> struct SumTraits<false> {
  using elem_t = rccl_float8;
  static __device__ fp8x2_storage_t packed(fp8x2_storage_t x, fp8x2_storage_t y) { return hadd2(x, y); }
  static __device__ elem_t          scalar(elem_t a, elem_t b)                   { return hadd(a, b); }
  static constexpr float            kMaxFinite = RCCL_FP8_MAX_FINITE;
};

template<> struct SumTraits<true> {
  using elem_t = rccl_bfloat8;
  static __device__ fp8x2_storage_t packed(fp8x2_storage_t x, fp8x2_storage_t y) { return hadd2_b(x, y); }
  static __device__ elem_t          scalar(elem_t a, elem_t b)                   { return hadd_b(a, b); }
  static constexpr float            kMaxFinite = RCCL_BF8_MAX_FINITE;
};

template<bool IsBf8>
__device__ inline void decode2(fp8x2_storage_t v, float& lo, float& hi) {
  union { typename SumTraits<IsBf8>::elem_t e[2]; fp8x2_storage_t s; } u;
  u.s = v;
  lo = float(u.e[0]);
  hi = float(u.e[1]);
}

// RCCL_FP8_MAX_FINITE differs between the host pass (OCP, 448) and an fnuz
// device build (240), so the device reports the limit it actually saturates to.
template<bool IsBf8>
__global__ void kReportMaxFinite(float* out) { *out = SumTraits<IsBf8>::kMaxFinite; }

template<bool IsBf8>
__global__ void kSum(const fp8x2_storage_t* __restrict__ X, const fp8x2_storage_t* __restrict__ Y,
                     fp8x2_storage_t* __restrict__ packedRaw, fp8x2_storage_t* __restrict__ scalarRaw,
                     fp8x2_storage_t* __restrict__ oracleRaw, float* __restrict__ packedF,
                     float* __restrict__ scalarF, float* __restrict__ oracleF,
                     float* __restrict__ aF, float* __restrict__ bF, int n) {
  using elem_t = typename SumTraits<IsBf8>::elem_t;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const fp8x2_storage_t x = X[i];
  const fp8x2_storage_t y = Y[i];

  union { elem_t e[2]; fp8x2_storage_t s; } ux, uy, us, uo;
  ux.s = x;
  uy.s = y;
  for (int l = 0; l < 2; ++l) {
    us.e[l] = SumTraits<IsBf8>::scalar(ux.e[l], uy.e[l]);
    uo.e[l] = elem_t(float(ux.e[l]) + float(uy.e[l]));
  }

  const fp8x2_storage_t packed = SumTraits<IsBf8>::packed(x, y);
  packedRaw[i] = packed;
  scalarRaw[i] = us.s;
  oracleRaw[i] = uo.s;
  decode2<IsBf8>(packed, packedF[2 * i], packedF[2 * i + 1]);
  decode2<IsBf8>(us.s, scalarF[2 * i], scalarF[2 * i + 1]);
  decode2<IsBf8>(uo.s, oracleF[2 * i], oracleF[2 * i + 1]);
  decode2<IsBf8>(x, aF[2 * i], aF[2 * i + 1]);
  decode2<IsBf8>(y, bF[2 * i], bF[2 * i + 1]);
}

class NarrowFloatSumTest : public DeviceTestBase {
protected:
  template<bool IsBf8>
  float deviceMaxFinite() {
    DeviceBuffer<float> d(1);
    kReportMaxFinite<IsBf8><<<1, 1>>>(d.ptr);
    syncAndCheck();
    return d.download();
  }

  // Bit-for-bit comparison. A mismatch where both sides decode to NaN is
  // reported separately: that is a difference in NaN encoding, not in value.
  static void expectBitMatch(const std::vector<fp8x2_storage_t>& got,
                             const std::vector<fp8x2_storage_t>& want,
                             const std::vector<float>& gotF, const std::vector<float>& wantF,
                             const char* what) {
    int mismatches = 0, nanEncodingOnly = 0, reported = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] == want[i]) continue;
      for (int l = 0; l < 2; ++l) {
        const uint8_t g = static_cast<uint8_t>((got[i] >> (8 * l)) & 0xFF);
        const uint8_t w = static_cast<uint8_t>((want[i] >> (8 * l)) & 0xFF);
        if (g == w) continue;
        if (std::isnan(gotF[2 * i + l]) && std::isnan(wantF[2 * i + l])) {
          ++nanEncodingOnly;
          continue;
        }
        if (reported++ < 10)
          ADD_FAILURE() << what << ": pair " << i << " lane " << l << " got 0x" << std::hex
                        << unsigned(g) << " (" << std::dec << gotF[2 * i + l] << ") want 0x"
                        << std::hex << unsigned(w) << " (" << std::dec << wantF[2 * i + l] << ")";
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << what << ": " << mismatches << " lanes differ in value";
    if (nanEncodingOnly != 0)
      GTEST_LOG_(INFO) << what << ": " << nanEncodingOnly
                       << " lanes are NaN on both sides with different encodings";
  }

  // The exact sum is computed in double, which is exact for these operands:
  // every fp8/bf8 value and every sum of two of them is representable there, so
  // the only rounding in play is the one the helper itself performs.
  static void expectRule(const std::vector<float>& aF, const std::vector<float>& bF,
                         const std::vector<float>& rF, float maxFinite, const char* what) {
    int badFinite = 0, badSat = 0, badSpecial = 0, reported = 0, saturating = 0;
    for (size_t k = 0; k < rF.size(); ++k) {
      const double a = aF[k], b = bF[k], r = rF[k];
      const double exact = a + b;

      if (std::isnan(a) || std::isnan(b) || std::isnan(exact)) {
        if (!std::isnan(r)) {
          if (reported++ < 10)
            ADD_FAILURE() << what << ": " << a << " + " << b << " = " << r << ", expected NaN";
          ++badSpecial;
        }
        continue;
      }

      if (std::isinf(a) || std::isinf(b)) {
        // Inf is expected, but a destination without an Inf encoding can only
        // answer NaN, so that is accepted too.
        const bool ok = (std::isinf(r) && std::signbit(r) == std::signbit(exact)) || std::isnan(r);
        if (!ok) {
          if (reported++ < 10)
            ADD_FAILURE() << what << ": " << a << " + " << b << " = " << r
                          << ", expected Inf of the same sign, or NaN";
          ++badSpecial;
        }
        continue;
      }

      if (!std::isfinite(r) || std::fabs(r) > maxFinite) {
        if (reported++ < 10)
          ADD_FAILURE() << what << ": " << a << " + " << b << " = " << exact << " came back as "
                        << r << ", expected a finite value within +/-" << maxFinite;
        ++badFinite;
        continue;
      }
      if (std::fabs(exact) > maxFinite) {
        ++saturating;
        if (r != std::copysign(maxFinite, exact)) {
          if (reported++ < 10)
            ADD_FAILURE() << what << ": " << a << " + " << b << " = " << exact << " came back as "
                          << r << ", expected " << std::copysign(maxFinite, exact);
          ++badSat;
        }
      }
    }
    EXPECT_EQ(badFinite, 0) << what << ": " << badFinite << " lanes left the finite range";
    EXPECT_EQ(badSat, 0) << what << ": " << badSat << " lanes did not saturate to the limit";
    EXPECT_EQ(badSpecial, 0) << what << ": " << badSpecial << " lanes mishandled Inf or NaN";
    // The overflow cases are the whole point, so fail rather than pass quietly
    // if the sweep somehow contains none of them.
    EXPECT_GT(saturating, 0) << what << ": no lane overflowed, so saturation went unexercised";
    GTEST_LOG_(INFO) << what << ": " << saturating << " of " << rF.size()
                     << " lanes overflowed and had to saturate";
  }

  // Every ordered pair of bytes. Lane 0 carries (a,b) and lane 1 the swapped
  // (b,a), so each lane independently sees all 65536 pairs and a lane-swap or
  // lane-drop bug cannot cancel out.
  template<bool IsBf8>
  void run(const char* what) {
    const int N = 256 * 256;
    std::vector<fp8x2_storage_t> hx(N), hy(N);
    for (int idx = 0; idx < N; ++idx) {
      const uint8_t a = static_cast<uint8_t>(idx & 0xFF);
      const uint8_t b = static_cast<uint8_t>((idx >> 8) & 0xFF);
      hx[idx] = static_cast<fp8x2_storage_t>(a | (static_cast<uint16_t>(b) << 8));
      hy[idx] = static_cast<fp8x2_storage_t>(b | (static_cast<uint16_t>(a) << 8));
    }
    DeviceBuffer<fp8x2_storage_t> dx(N), dy(N), dp(N), ds(N), dor(N);
    dx.copyFrom(hx);
    dy.copyFrom(hy);
    DeviceBuffer<float> dpf(2 * N), dsf(2 * N), dof(2 * N), daf(2 * N), dbf(2 * N);

    kSum<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(dx.ptr, dy.ptr, dp.ptr, ds.ptr, dor.ptr, dpf.ptr,
                                                   dsf.ptr, dof.ptr, daf.ptr, dbf.ptr, N);
    syncAndCheck();

    const std::vector<fp8x2_storage_t> packedRaw = dp.copyTo(), scalarRaw = ds.copyTo(),
                                       oracleRaw = dor.copyTo();
    const std::vector<float> packedF = dpf.copyTo(), scalarF = dsf.copyTo(), oracleF = dof.copyTo();
    const std::vector<float> aF = daf.copyTo(), bF = dbf.copyTo();

    // Checking both widths against the oracle also settles them against each
    // other, so there is no third comparison.
    const std::string two = std::string(what) + " 2-wide vs float() oracle";
    const std::string one = std::string(what) + " 1-wide vs float() oracle";
    expectBitMatch(packedRaw, oracleRaw, packedF, oracleF, two.c_str());
    expectBitMatch(scalarRaw, oracleRaw, scalarF, oracleF, one.c_str());
    expectRule(aF, bF, packedF, deviceMaxFinite<IsBf8>(), what);
  }
};

TEST_F(NarrowFloatSumTest, Fp8Sum) { run<false>("hadd2"); }
TEST_F(NarrowFloatSumTest, Bf8Sum) { run<true>("hadd2_b"); }

} // namespace RcclUnitTesting
