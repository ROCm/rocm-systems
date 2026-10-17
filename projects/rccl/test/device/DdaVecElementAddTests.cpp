/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// vecElementAdd<T> adds a[k] + b[k] per element in memory order, bit-exact with a host reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/device/CollCommon.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

template <typename T>
__global__ void VecElementAddKernel(const uint4* a, const uint4* b, uint4* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = dda::common::vecElementAdd<T>(a[i], b[i]);
  }
}

template <typename E>
struct VecAddCase {
  E a;
  E b;
  E expected;
};

uint32_t FloatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

float BitsToFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

bool IsHalfNan(uint16_t u) {
  return (u & 0x7C00u) == 0x7C00u && (u & 0x03FFu) != 0;
}

bool IsBf16Nan(uint16_t u) {
  return (u & 0x7F80u) == 0x7F80u && (u & 0x007Fu) != 0;
}

// fp32 sum rounded once to fp16; double rounding through fp32 is exact for fp16 since 24 >= 2 * 11 + 2.
uint16_t HalfAddRef(uint16_t a, uint16_t b) {
  _Float16 x;
  _Float16 y;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  const _Float16 r = static_cast<_Float16>(static_cast<float>(x) + static_cast<float>(y));
  uint16_t u;
  std::memcpy(&u, &r, sizeof(u));
  return u;
}

// fp32 sum rounded to bf16 with round-to-nearest-even, independent of the HIP bf16 library.
uint16_t Bf16AddRef(uint16_t a, uint16_t b) {
  const float s = BitsToFloat(static_cast<uint32_t>(a) << 16) + BitsToFloat(static_cast<uint32_t>(b) << 16);
  const uint32_t u = FloatBits(s);
  if (std::isnan(s)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Deterministic bit patterns, so a failure reproduces exactly.
template <typename E>
std::vector<E> PseudoRandomBits(size_t n, uint32_t seed) {
  std::vector<E> v(n);
  uint32_t state = seed;
  for (size_t i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    v[i] = static_cast<E>(state >> (32 - 8 * sizeof(E)));
  }
  return v;
}

} // namespace

class DdaVecElementAdd : public DeviceTestBase {
protected:
  // Runs vecElementAdd<T> over element arrays a and b viewed as uint4, zero-padding the shorter one and the tail.
  template <typename T, typename E>
  std::vector<E> Run(const std::vector<E>& a, const std::vector<E>& b) {
    static_assert(sizeof(T) == sizeof(E), "element storage must match T");
    constexpr size_t kPerVec = sizeof(uint4) / sizeof(E);
    const size_t nElems = std::max(a.size(), b.size());
    const size_t nVec = (nElems + kPerVec - 1) / kPerVec;
    std::vector<uint4> ha(nVec, uint4{0, 0, 0, 0});
    std::vector<uint4> hb(nVec, uint4{0, 0, 0, 0});
    std::memcpy(ha.data(), a.data(), a.size() * sizeof(E));
    std::memcpy(hb.data(), b.data(), b.size() * sizeof(E));

    DeviceBuffer<uint4> dA(nVec);
    DeviceBuffer<uint4> dB(nVec);
    DeviceBuffer<uint4> dOut(nVec);
    dA.copyFrom(ha);
    dB.copyFrom(hb);
    dOut.zero();
    const int n = static_cast<int>(nVec);
    VecElementAddKernel<T><<<gridFor(nVec), kDefaultBlockSize>>>(dA.ptr, dB.ptr, dOut.ptr, n);
    syncAndCheck();

    const std::vector<uint4> hOut = dOut.copyTo();
    std::vector<E> out(nElems);
    std::memcpy(out.data(), hOut.data(), nElems * sizeof(E));
    return out;
  }

  template <typename T, typename E>
  void ExpectCases(const std::vector<VecAddCase<E>>& cases) {
    std::vector<E> a;
    std::vector<E> b;
    for (const auto& c : cases) {
      a.push_back(c.a);
      b.push_back(c.b);
    }
    const std::vector<E> out = Run<T>(a, b);
    for (size_t i = 0; i < cases.size(); ++i) {
      EXPECT_EQ(out[i], cases[i].expected)
        << std::hex << "element " << std::dec << i << " a=0x" << std::hex << +cases[i].a << " b=0x" << +cases[i].b;
    }
  }
};

TEST_F(DdaVecElementAdd, FloatCraftedLanes) {
  const float kInf = std::numeric_limits<float>::infinity();
  // The last pair is the 2^24 + 1 tie, which rounds to even.
  const std::vector<std::pair<float, float>> in = {{1.0f, 2.0f},    {0.0f, -0.0f}, {-0.0f, -0.0f}, {kInf, 1.0f},
                                                   {1e30f, -1e30f}, {0.5f, 0.25f}, {-3.0f, 1.0f},  {16777216.0f, 1.0f}};
  const std::vector<float> expected = {3.0f, 0.0f, -0.0f, kInf, 0.0f, 0.75f, -2.0f, 16777216.0f};
  std::vector<VecAddCase<uint32_t>> cases;
  for (size_t i = 0; i < in.size(); ++i) {
    cases.push_back({FloatBits(in[i].first), FloatBits(in[i].second), FloatBits(expected[i])});
  }
  ExpectCases<float>(cases);
}

TEST_F(DdaVecElementAdd, FloatMatchesHostReference) {
  const auto a = PseudoRandomBits<uint32_t>(4096, 1);
  const auto b = PseudoRandomBits<uint32_t>(4096, 2);
  const auto out = Run<float>(a, b);
  for (size_t i = 0; i < a.size(); ++i) {
    const float ref = BitsToFloat(a[i]) + BitsToFloat(b[i]);
    if (std::isnan(ref)) {
      ASSERT_TRUE(std::isnan(BitsToFloat(out[i]))) << "element " << i;
    } else {
      ASSERT_EQ(out[i], FloatBits(ref)) << "element " << i;
    }
  }
}

TEST_F(DdaVecElementAdd, HalfCraftedLanes) {
  ExpectCases<half, uint16_t>({
    {0x3C00, 0x4000, 0x4200},  // 1 + 2 = 3
    {0x4400, 0x3C00, 0x4500},  // 4 + 1 = 5
    {0x0000, 0x8000, 0x0000},  // +0 + -0 = +0
    {0x8000, 0x8000, 0x8000},  // -0 + -0 = -0
    {0x7C00, 0x3C00, 0x7C00},  // inf + 1 = inf
    {0x0001, 0x0001, 0x0002},  // smallest subnormals
    {0x03FF, 0x0001, 0x0400},  // largest subnormal carries into the smallest normal
    {0x6800, 0x3C00, 0x6800},  // 2048 + 1 ties to even 2048
    {0x6801, 0x3C00, 0x6802},  // 2050 + 1 ties to even 2052
    {0x7BFF, 0x5000, 0x7C00},  // 65504 + 32 overflows to inf
    {0xC000, 0x3C00, 0xBC00},  // -2 + 1 = -1
  });
}

TEST_F(DdaVecElementAdd, HalfMatchesHostReference) {
  const auto a = PseudoRandomBits<uint16_t>(8192, 3);
  const auto b = PseudoRandomBits<uint16_t>(8192, 4);
  const auto out = Run<half>(a, b);
  for (size_t i = 0; i < a.size(); ++i) {
    const uint16_t ref = HalfAddRef(a[i], b[i]);
    if (IsHalfNan(ref)) {
      ASSERT_TRUE(IsHalfNan(out[i])) << "element " << i;
    } else {
      ASSERT_EQ(out[i], ref) << std::hex << "element " << std::dec << i << " a=0x" << std::hex << a[i] << " b=0x"
                             << b[i];
    }
  }
}

TEST_F(DdaVecElementAdd, Bf16CraftedLanes) {
  ExpectCases<bf16, uint16_t>({
    {0x3F80, 0x4000, 0x4040},  // 1 + 2 = 3
    {0x4080, 0x3F80, 0x40A0},  // 4 + 1 = 5
    {0x0000, 0x8000, 0x0000},  // +0 + -0 = +0
    {0x8000, 0x8000, 0x8000},  // -0 + -0 = -0
    {0x7F80, 0x3F80, 0x7F80},  // inf + 1 = inf
    {0x3F80, 0x3B80, 0x3F80},  // 1 + 2^-8 ties to even 1
    {0x3F81, 0x3B80, 0x3F82},  // (1 + 2^-7) + 2^-8 ties to even 1 + 2^-6; truncation gives 0x3F81
    {0x3F80, 0x3BC0, 0x3F81},  // 1 + 1.5 * 2^-8 is above the half ulp and rounds up
    {0x4780, 0x4780, 0x4800},  // 65536 + 65536 keeps bf16's 8-bit exponent; fp16 would overflow
    {0x7F7F, 0x7F7F, 0x7F80},  // max finite doubled overflows to inf
    {0xC000, 0x3F80, 0xBF80},  // -2 + 1 = -1
  });
}

TEST_F(DdaVecElementAdd, Bf16MatchesHostReference) {
  const auto a = PseudoRandomBits<uint16_t>(8192, 5);
  const auto b = PseudoRandomBits<uint16_t>(8192, 6);
  const auto out = Run<bf16>(a, b);
  for (size_t i = 0; i < a.size(); ++i) {
    const uint16_t ref = Bf16AddRef(a[i], b[i]);
    if (IsBf16Nan(ref)) {
      ASSERT_TRUE(IsBf16Nan(out[i])) << "element " << i;
    } else {
      ASSERT_EQ(out[i], ref) << std::hex << "element " << std::dec << i << " a=0x" << std::hex << a[i] << " b=0x"
                             << b[i];
    }
  }
}

} // namespace RcclUnitTesting
