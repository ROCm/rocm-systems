/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the fp8/bf8 reduce helpers in src/include/rccl_float8.h.
//
// These operate on fp8x2_storage_t, a uint16_t holding two fp8 bytes (element 0
// in the low byte). Depending on the target they take either an
// architecture-specific packed path or a per-element fallback.
//
// The invariant under test: the 2-wide helper is bit-identical to applying the
// 1-wide helper to each element. On gfx942 that is what catches the scalar
// hadd/hadd_b defect, where the packed bytes returned by
// __builtin_amdgcn_cvt_pk_{fp8,bf8}_f32 were fed to rccl_float8's numeric int
// constructor and converted as a value instead of reinterpreted as storage.
//
// The oracle runs on the device rather than the host because the fp8 encoding is
// not portable: gfx942 and the software fallback use the fnuz variants (e4m3 max
// 240) while gfx950 and gfx12xx use OCP (448), so a host-computed reference
// would disagree with the device for reasons unrelated to the helpers. Results
// come back as raw bytes and are compared bit-for-bit; decoded floats come back
// too, only to make failures readable and to tell a genuine numeric difference
// apart from a NaN that is merely encoded differently.

#include "DeviceTestBase.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "rccl_float8.h"

namespace RcclUnitTesting
{

// Compile-time selection between the fp8 (e4m3) and bf8 (e5m2) helper family.
template<bool IsBf8> struct NarrowTraits;

template<> struct NarrowTraits<false> {
  using elem_t = rccl_float8;
  static __device__ fp8x2_storage_t add2(fp8x2_storage_t x, fp8x2_storage_t y) { return hadd2(x, y); }
  static __device__ elem_t          add(elem_t a, elem_t b)                    { return hadd(a, b); }
};

template<> struct NarrowTraits<true> {
  using elem_t = rccl_bfloat8;
  static __device__ fp8x2_storage_t add2(fp8x2_storage_t x, fp8x2_storage_t y) { return hadd2_b(x, y); }
  static __device__ elem_t          add(elem_t a, elem_t b)                    { return hadd_b(a, b); }
};

template<bool IsBf8>
__device__ inline void decode2(fp8x2_storage_t v, float& lo, float& hi) {
  union { typename NarrowTraits<IsBf8>::elem_t e[2]; fp8x2_storage_t s; } u;
  u.s = v;
  lo = float(u.e[0]);
  hi = float(u.e[1]);
}

template<bool IsBf8>
__global__ void kNarrowAdd(const fp8x2_storage_t* __restrict__ X,
                           const fp8x2_storage_t* __restrict__ Y,
                           fp8x2_storage_t* __restrict__ packedRaw,
                           fp8x2_storage_t* __restrict__ refRaw,
                           float* __restrict__ packedF,
                           float* __restrict__ refF, int n) {
  using elem_t = typename NarrowTraits<IsBf8>::elem_t;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  fp8x2_storage_t x = X[i], y = Y[i];
  fp8x2_storage_t packed = NarrowTraits<IsBf8>::add2(x, y);

  union { elem_t e[2]; fp8x2_storage_t s; } ux, uy, uw;
  ux.s = x;
  uy.s = y;
  uw.e[0] = NarrowTraits<IsBf8>::add(ux.e[0], uy.e[0]);
  uw.e[1] = NarrowTraits<IsBf8>::add(ux.e[1], uy.e[1]);

  packedRaw[i] = packed;
  refRaw[i]    = uw.s;
  decode2<IsBf8>(packed, packedF[2 * i], packedF[2 * i + 1]);
  decode2<IsBf8>(uw.s, refF[2 * i], refF[2 * i + 1]);
}

class NarrowFloatReduceTest : public DeviceTestBase {
protected:
  // Bit-for-bit comparison. A mismatch where both lanes decode to NaN is
  // reported separately: that is a difference in NaN encoding, not in value.
  static void expectBitMatch(const std::vector<fp8x2_storage_t>& packedRaw,
                             const std::vector<fp8x2_storage_t>& refRaw,
                             const std::vector<float>& packedF,
                             const std::vector<float>& refF,
                             const char* name) {
    int valueMismatches = 0;
    int nanEncodingOnly = 0;
    for (size_t i = 0; i < packedRaw.size(); ++i) {
      if (packedRaw[i] == refRaw[i]) continue;
      for (int l = 0; l < 2; ++l) {
        uint8_t pb = static_cast<uint8_t>((packedRaw[i] >> (8 * l)) & 0xFF);
        uint8_t rb = static_cast<uint8_t>((refRaw[i] >> (8 * l)) & 0xFF);
        if (pb == rb) continue;
        float p = packedF[2 * i + l], r = refF[2 * i + l];
        if (std::isnan(p) && std::isnan(r)) {
          ++nanEncodingOnly;
          continue;
        }
        if (valueMismatches < 10)
          ADD_FAILURE() << name << ": pair " << i << " lane " << l << " packed=0x" << std::hex
                        << unsigned(pb) << " (" << std::dec << p << ") scalar=0x" << std::hex
                        << unsigned(rb) << " (" << std::dec << r << ")";
        ++valueMismatches;
      }
    }
    EXPECT_EQ(valueMismatches, 0) << name << ": " << valueMismatches << " lanes differ in value";
    if (nanEncodingOnly != 0)
      GTEST_LOG_(INFO) << name << ": " << nanEncodingOnly
                       << " lanes are NaN in both but with different encodings";
  }

  // Every ordered pair of fp8 bytes. Lane 0 carries (a,b) and lane 1 the
  // swapped (b,a), so each lane independently sees all 65536 pairs and a
  // lane-swap bug cannot cancel out.
  template<bool IsBf8>
  void runAdd(const char* name) {
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

    kNarrowAdd<IsBf8><<<gridFor(N), kDefaultBlockSize>>>(dx.ptr, dy.ptr, dpr.ptr, drr.ptr,
                                                         dpf.ptr, drf.ptr, N);
    syncAndCheck();
    expectBitMatch(dpr.copyTo(), drr.copyTo(), dpf.copyTo(), drf.copyTo(), name);
  }
};

TEST_F(NarrowFloatReduceTest, Fp8Add) { runAdd<false>("hadd2 vs hadd"); }
TEST_F(NarrowFloatReduceTest, Bf8Add) { runAdd<true>("hadd2_b vs hadd_b"); }

} // namespace RcclUnitTesting
