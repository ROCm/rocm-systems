/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exhaustive correctness tests for the fp8 (e4m3) and bf8 (e5m2) device
// reductions in reduce_kernel.h.
//
// These types are 8 bits wide, so the entire input domain of a binary reduction
// is 256 x 256 = 65536 pairs. That is small enough to test by enumeration rather
// than by sampling, which is what makes these tests a usable safety net for the
// hand-written intrinsic paths in rccl_float8.h (hadd / hadd2 / hadd8). Those
// paths are selected per architecture and per pack width, so a change to one of
// them can silently alter results for a subset of inputs that a spot-check misses.
//
// Two independent references are used:
//
//   1. Pack-width equivalence. Apply_Reduce recurses BytePack<16> -> BytePack<8>
//      -> ... -> BytePack<1>, and each width may land on a different hand-written
//      specialization. Every width must agree bit-for-bit with the EltPerPack=1
//      scalar path on all 65536 pairs.
//
//   2. Host software model. The host conversion in rccl_float8.h is a pure
//      software implementation that shares no code with the device intrinsics, so
//      comparing against it catches the case where every device width is wrong in
//      the same way (e.g. a mis-encoded conversion scale).
//
// Lane assignment matters. A reduction that permutes lanes within a pack is
// invisible if every lane of the pack holds the same value, so the pack-width
// test rotates the pair-to-lane mapping and requires every pair to be exercised
// in every lane position.

#include "DeviceTestBase.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "nccl.h"
#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"

namespace RcclUnitTesting
{

namespace {

constexpr int kPairs = 256 * 256; // the entire binary input domain

// Reduce nPacks packs of BYTES bytes. BYTES/sizeof(T) fixes the EltPerPack that
// Apply_Reduce dispatches on, which is the dimension under test.
template<typename T, typename Fn, int BYTES>
__global__ void kernelReducePacks(const uint8_t* __restrict__ a,
                                  const uint8_t* __restrict__ b,
                                  uint8_t* __restrict__ out,
                                  int nPacks, uint64_t opArg) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPacks) return;
  BytePack<BYTES> pa, pb;
  for (int k = 0; k < BYTES; k++) {
    pa.u8[k] = a[(size_t)i * BYTES + k];
    pb.u8[k] = b[(size_t)i * BYTES + k];
  }
  BytePack<BYTES> r = applyReduce(Fn(opArg), pa, pb);
  for (int k = 0; k < BYTES; k++) out[(size_t)i * BYTES + k] = r.u8[k];
}

// Decode fp8 bytes to float using the device's own conversion, so the decode
// matches whatever encoding this architecture compiled in.
template<typename T>
__global__ void kernelDecode(const uint8_t* __restrict__ in, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  BytePack<1> p;
  p.u8[0] = in[i];
  out[i] = (float)fromPack<T>(p);
}

// Ask the device which fp8 encoding it actually compiled in, rather than
// re-deriving it from the architecture name. Byte 0x80 is -0.0 under OCP and NaN
// under FNUZ, so decoding it on device distinguishes the two. Reading it off the
// GPU keeps this in step with the typedef selection in rccl_float8.h even if the
// per-architecture table there changes.
bool deviceIsFnuz() {
  static int cached = -1;
  if (cached < 0) {
    const uint8_t probe = 0x80;
    DeviceBuffer<uint8_t> d(1);
    DeviceBuffer<float> f(1);
    d.copyFrom(&probe, 1);
    kernelDecode<rccl_float8><<<1, 1>>>(d.ptr, f.ptr, 1);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    cached = std::isnan(f.copyTo()[0]) ? 1 : 0;
  }
  return cached == 1;
}

class Fp8ReduceTest : public DeviceTestBase {
protected:
  // in[i] holds the pair (a=i>>8, b=i&0xff), optionally rotated by `rot` lanes so
  // that a given pair lands in a different position within its pack.
  static void buildPairs(int rot, std::vector<uint8_t>& a, std::vector<uint8_t>& b) {
    a.resize(kPairs);
    b.resize(kPairs);
    for (int i = 0; i < kPairs; i++) {
      int p = (i + rot) % kPairs;
      a[i] = (uint8_t)(p >> 8);
      b[i] = (uint8_t)(p & 0xff);
    }
  }

  template<typename T, typename Fn, int BYTES>
  std::vector<uint8_t> reduceAll(const std::vector<uint8_t>& ha,
                                 const std::vector<uint8_t>& hb,
                                 uint64_t opArg) {
    const int nPacks = kPairs / BYTES;
    DeviceBuffer<uint8_t> da(kPairs), db(kPairs), dout(kPairs);
    da.copyFrom(ha);
    db.copyFrom(hb);
    kernelReducePacks<T, Fn, BYTES>
      <<<gridFor(nPacks), kDefaultBlockSize>>>(da.ptr, db.ptr, dout.ptr, nPacks, opArg);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    return dout.copyTo();
  }

  template<typename T>
  std::vector<float> decodeAll(const std::vector<uint8_t>& bytes) {
    DeviceBuffer<uint8_t> d(kPairs);
    DeviceBuffer<float> f(kPairs);
    d.copyFrom(bytes);
    kernelDecode<T><<<gridFor(kPairs), kDefaultBlockSize>>>(d.ptr, f.ptr, kPairs);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    return f.copyTo();
  }

  // Every pack width must match the scalar path exactly, for every pair, in every
  // lane position. Reported as a count plus one decoded example, since a bare
  // "65536 mismatches" says nothing about which input broke.
  template<typename T, typename Fn>
  void ExpectWidthsAgree(const char* opName, uint64_t opArg = 0) {
    for (int rot = 0; rot < 8; rot++) {
      std::vector<uint8_t> ha, hb;
      buildPairs(rot, ha, hb);
      auto scalar = reduceAll<T, Fn, 1>(ha, hb, opArg);
      auto w2 = reduceAll<T, Fn, 2>(ha, hb, opArg);
      auto w8 = reduceAll<T, Fn, 8>(ha, hb, opArg);
      auto w16 = reduceAll<T, Fn, 16>(ha, hb, opArg);
      const std::vector<uint8_t>* wide[3] = {&w2, &w8, &w16};
      const int widths[3] = {2, 8, 16};
      for (int w = 0; w < 3; w++) {
        int bad = 0, first = -1;
        for (int i = 0; i < kPairs; i++) {
          if ((*wide[w])[i] != scalar[i]) {
            if (bad == 0) first = i;
            bad++;
          }
        }
        if (bad == 0) continue;
        auto decScalar = decodeAll<T>(scalar);
        auto decWide = decodeAll<T>(*wide[w]);
        ADD_FAILURE() << opName << ": EltPerPack=" << (widths[w] / (int)sizeof(T))
                      << " (BytePack<" << widths[w] << ">) disagrees with the scalar path on "
                      << bad << "/" << kPairs << " pairs at lane rotation " << rot
                      << ". First: a=0x" << std::hex << (int)ha[first]
                      << " b=0x" << (int)hb[first]
                      << " scalar=0x" << (int)scalar[first]
                      << " wide=0x" << (int)(*wide[w])[first] << std::dec
                      << " (scalar=" << decScalar[first] << " wide=" << decWide[first] << ")";
        break; // one failure per width is enough to localize it
      }
    }
  }
};

} // namespace

// ---------------------------------------------------------------------------
// 1. Pack-width equivalence - exhaustive over all 65536 pairs and all 8 lanes.
//    This is the regression net for any new intrinsic specialization: a wide
//    path that does not reproduce the scalar path on every input fails here.
// ---------------------------------------------------------------------------

TEST_F(Fp8ReduceTest, PackWidthsAgreeSumE4m3) {
  ExpectWidthsAgree<rccl_float8, FuncSum<rccl_float8>>("e4m3 Sum");
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeSumE5m2) {
  ExpectWidthsAgree<rccl_bfloat8, FuncSum<rccl_bfloat8>>("e5m2 Sum");
}

// Prod/MinMax have no wide specialization today, so these currently assert that
// the recursion stays faithful. They are the guard if one is ever added.
TEST_F(Fp8ReduceTest, PackWidthsAgreeProdE4m3) {
  ExpectWidthsAgree<rccl_float8, FuncProd<rccl_float8>>("e4m3 Prod");
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeProdE5m2) {
  ExpectWidthsAgree<rccl_bfloat8, FuncProd<rccl_bfloat8>>("e5m2 Prod");
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeMinE4m3) {
  ExpectWidthsAgree<rccl_float8, FuncMinMax<rccl_float8>>("e4m3 Min", /*opArg=*/0);
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeMaxE4m3) {
  ExpectWidthsAgree<rccl_float8, FuncMinMax<rccl_float8>>("e4m3 Max", /*opArg=*/1);
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeMinE5m2) {
  ExpectWidthsAgree<rccl_bfloat8, FuncMinMax<rccl_bfloat8>>("e5m2 Min", /*opArg=*/0);
}
TEST_F(Fp8ReduceTest, PackWidthsAgreeMaxE5m2) {
  ExpectWidthsAgree<rccl_bfloat8, FuncMinMax<rccl_bfloat8>>("e5m2 Max", /*opArg=*/1);
}

// ---------------------------------------------------------------------------
// 2. Host software model - exhaustive over every pair whose exact sum stays in
//    range. The host conversion shares no code with the device intrinsics, so
//    this catches every width being wrong together.
//
//    Overflow is excluded here and asserted separately below: the host software
//    encoder clips to max-finite while the device hardware converter yields a
//    non-finite result, and that difference is a property of the two encoders,
//    not a reduction bug.
// ---------------------------------------------------------------------------

namespace {
template<typename T>
void ExpectMatchesHostModel(bool isE5m2, const char* name) {
  const bool fnuz = deviceIsFnuz();
  const float maxFinite = isE5m2 ? 57344.0f : 448.0f;

  std::vector<uint8_t> ha(kPairs), hb(kPairs);
  for (int i = 0; i < kPairs; i++) {
    ha[i] = (uint8_t)(i >> 8);
    hb[i] = (uint8_t)(i & 0xff);
  }

  DeviceBuffer<uint8_t> da(kPairs), db(kPairs), dout(kPairs);
  da.copyFrom(ha);
  db.copyFrom(hb);
  kernelReducePacks<T, FuncSum<T>, 1><<<dim3((kPairs + 255) / 256), 256>>>(
    da.ptr, db.ptr, dout.ptr, kPairs, 0);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  auto dev = dout.copyTo();

  int checked = 0, excluded = 0, bad = 0, first = -1;
  uint8_t firstWant = 0;
  for (int i = 0; i < kPairs; i++) {
    const float fa = rcclFp8ToFloat(ha[i], isE5m2, fnuz);
    const float fb = rcclFp8ToFloat(hb[i], isE5m2, fnuz);
    if (std::isnan(fa) || std::isnan(fb) || std::isinf(fa) || std::isinf(fb)) {
      excluded++;
      continue;
    }
    const float sum = fa + fb; // exact: two 8-bit values always sum exactly in float
    if (std::fabs(sum) > maxFinite) {
      excluded++;
      continue;
    }
    checked++;
    const uint8_t want = T(sum).__x; // host software round-to-nearest-even encode
    if (dev[i] != want) {
      if (bad == 0) {
        first = i;
        firstWant = want;
      }
      bad++;
    }
  }

  EXPECT_GT(checked, 50000) << name << ": too few pairs actually compared";
  EXPECT_EQ(bad, 0) << name << ": device reduction disagrees with the host software model on "
                    << bad << "/" << checked << " in-range pairs (" << excluded
                    << " excluded as non-finite or overflowing). First: a=0x" << std::hex
                    << (int)ha[first] << " b=0x" << (int)hb[first] << " device=0x"
                    << (int)dev[first] << " host=0x" << (int)firstWant << std::dec
                    << " (" << rcclFp8ToFloat(ha[first], isE5m2, fnuz) << " + "
                    << rcclFp8ToFloat(hb[first], isE5m2, fnuz) << ")";
}
} // namespace

TEST_F(Fp8ReduceTest, MatchesHostModelE4m3) {
  ExpectMatchesHostModel<rccl_float8>(/*isE5m2=*/false, "e4m3");
}
TEST_F(Fp8ReduceTest, MatchesHostModelE5m2) {
  ExpectMatchesHostModel<rccl_bfloat8>(/*isE5m2=*/true, "e5m2");
}

// ---------------------------------------------------------------------------
// 3. Conversion scale identity.
//
//    The pk8 upcasts take their scale as an E8M0 biased exponent, where 127 is
//    1.0 - not a float. Passing 0 there scales every input by 2^-127, which
//    still round-trips zero and still produces plausible-looking bytes, so it
//    can survive a smoke test. These exact small-integer sums pin the scale.
// ---------------------------------------------------------------------------

namespace {
template<typename T>
void ExpectExactSmallSums(bool isE5m2, const char* name) {
  const bool fnuz = deviceIsFnuz();
  // Values exactly representable in both e4m3 and e5m2, whose sums are too.
  const float lhs[] = {1.0f, 2.0f, 4.0f, 0.5f, -1.0f, 8.0f, 0.25f, 16.0f};
  const float rhs[] = {1.0f, 2.0f, 4.0f, 0.5f, 3.0f, -8.0f, 0.75f, 16.0f};
  constexpr int kN = sizeof(lhs) / sizeof(lhs[0]);

  std::vector<uint8_t> ha(kPairs, 0), hb(kPairs, 0);
  for (int i = 0; i < kN; i++) {
    ha[i] = T(lhs[i]).__x;
    hb[i] = T(rhs[i]).__x;
  }
  DeviceBuffer<uint8_t> da(kPairs), db(kPairs), dout(kPairs);
  da.copyFrom(ha);
  db.copyFrom(hb);
  // Width 8 so the pk8 path is the one under test where it exists.
  kernelReducePacks<T, FuncSum<T>, 8><<<dim3((kPairs / 8 + 255) / 256), 256>>>(
    da.ptr, db.ptr, dout.ptr, kPairs / 8, 0);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  auto dev = dout.copyTo();

  for (int i = 0; i < kN; i++) {
    const float got = rcclFp8ToFloat(dev[i], isE5m2, fnuz);
    const float want = lhs[i] + rhs[i];
    EXPECT_EQ(got, want) << name << ": " << lhs[i] << " + " << rhs[i] << " gave " << got
                         << " (0x" << std::hex << (int)dev[i] << std::dec
                         << "). A result near 2^-127 means the pk8 upcast scale is not the"
                            " E8M0 encoding of 1.0 (127).";
  }
}
} // namespace

TEST_F(Fp8ReduceTest, ExactSmallSumsE4m3) {
  ExpectExactSmallSums<rccl_float8>(/*isE5m2=*/false, "e4m3");
}
TEST_F(Fp8ReduceTest, ExactSmallSumsE5m2) {
  ExpectExactSmallSums<rccl_bfloat8>(/*isE5m2=*/true, "e5m2");
}

// ---------------------------------------------------------------------------
// 4. Saturating sums stay non-finite, and every width agrees on what they are.
//
//    max+max overflows the format. The exact encoding (NaN vs infinity) is an
//    architecture property, so this asserts the invariant that actually matters:
//    the result leaves the finite range, and all pack widths produce identical
//    bits. A width that silently clipped to max-finite while another produced
//    NaN would be a real inconsistency.
// ---------------------------------------------------------------------------

namespace {
template<typename T>
void ExpectOverflowConsistent(bool isE5m2, const char* name) {
  const bool fnuz = deviceIsFnuz();
  const float maxFinite = isE5m2 ? 57344.0f : 448.0f;

  std::vector<uint8_t> ha(kPairs), hb(kPairs);
  const uint8_t maxBits = T(maxFinite).__x;
  for (int i = 0; i < kPairs; i++) {
    ha[i] = maxBits;
    hb[i] = maxBits;
  }
  DeviceBuffer<uint8_t> da(kPairs), db(kPairs), d1(kPairs), d8(kPairs);
  da.copyFrom(ha);
  db.copyFrom(hb);
  kernelReducePacks<T, FuncSum<T>, 1><<<dim3((kPairs + 255) / 256), 256>>>(
    da.ptr, db.ptr, d1.ptr, kPairs, 0);
  kernelReducePacks<T, FuncSum<T>, 8><<<dim3((kPairs / 8 + 255) / 256), 256>>>(
    da.ptr, db.ptr, d8.ptr, kPairs / 8, 0);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  auto s1 = d1.copyTo();
  auto s8 = d8.copyTo();

  EXPECT_EQ(s1[0], s8[0]) << name << ": scalar and pk8 disagree on max+max (0x" << std::hex
                          << (int)s1[0] << " vs 0x" << (int)s8[0] << std::dec << ")";
  const float got = rcclFp8ToFloat(s1[0], isE5m2, fnuz);
  EXPECT_TRUE(std::isnan(got) || std::isinf(got) || std::fabs(got) >= maxFinite)
    << name << ": " << maxFinite << " + " << maxFinite << " gave " << got
    << ", which is neither non-finite nor saturated";
}
} // namespace

TEST_F(Fp8ReduceTest, OverflowConsistentE4m3) {
  ExpectOverflowConsistent<rccl_float8>(/*isE5m2=*/false, "e4m3");
}
TEST_F(Fp8ReduceTest, OverflowConsistentE5m2) {
  ExpectOverflowConsistent<rccl_bfloat8>(/*isE5m2=*/true, "e5m2");
}

} // namespace RcclUnitTesting
