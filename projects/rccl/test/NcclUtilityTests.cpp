/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side tests for nccl::utility (nccl_device/utility.h). bitops.h has same-named copies, so calls are qualified.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "gtest/gtest.h"

// hip_compat.h maps NCCL_DEVICE_INLINE to __device__ __forceinline__, which hip_runtime.h must define first.
#include "nccl_device/utility.h"

namespace RcclUnitTesting {
namespace {

namespace nu = nccl::utility;

// xorshift64: deterministic sweep inputs without <random> distribution differences across libstdc++ versions.
class XorShift64 {
public:
  explicit XorShift64(uint64_t seed) : s_(seed) {}
  uint64_t next() {
    s_ ^= s_ << 13;
    s_ ^= s_ >> 7;
    s_ ^= s_ << 17;
    return s_;
  }

private:
  uint64_t s_;
};

// Schoolbook high word from half-width limbs: shares no expression with utility.h's wide-multiply host branch.
template <typename U, typename Half, int kHalfBits>
U refMulHi(U a, U b) {
  const U mask = static_cast<U>(static_cast<Half>(~Half(0)));
  const U aLo = a & mask;
  const U aHi = a >> kHalfBits;
  const U bLo = b & mask;
  const U bHi = b >> kHalfBits;
  const U lolo = aLo * bLo;
  const U lohi = aLo * bHi;
  const U hilo = aHi * bLo;
  const U mid = (lolo >> kHalfBits) + (lohi & mask) + (hilo & mask);
  return aHi * bHi + (lohi >> kHalfBits) + (hilo >> kHalfBits) + (mid >> kHalfBits);
}

uint32_t refMul32hi(uint32_t a, uint32_t b) {
  return refMulHi<uint32_t, uint16_t, 16>(a, b);
}

uint64_t refMul64hi(uint64_t a, uint64_t b) {
  return refMulHi<uint64_t, uint32_t, 32>(a, b);
}

bool refIsPow2(uint64_t y) {
  while (y > 1 && (y % 2) == 0) {
    y /= 2;
  }
  return y == 1;
}

// Defining property of idivRcp: rcp * y <= 2^bits - 1 < (rcp + 1) * y, except a power of 2 gets 2^bits / y (mod 2^bits).
template <int kBits>
void expectRcpProperty(uint64_t y, uint64_t rcp) {
  const unsigned __int128 two = static_cast<unsigned __int128>(1) << kBits;
  if (refIsPow2(y)) {
    EXPECT_EQ((static_cast<unsigned __int128>(rcp) + (y == 1 ? two : 0)) * y, two) << "y=" << y << " rcp=" << rcp;
  } else {
    EXPECT_LE(static_cast<unsigned __int128>(rcp) * y, two - 1) << "y=" << y << " rcp=" << rcp;
    EXPECT_GT((static_cast<unsigned __int128>(rcp) + 1) * y, two - 1) << "y=" << y << " rcp=" << rcp;
  }
}

// Divisors that hit every reciprocal regime: y=1 (rcp wraps to 0), powers of 2 (+1 term), odd, and near-max.
const std::vector<uint32_t>& divisors32() {
  static const std::vector<uint32_t> kDivs = {1u,          2u,          3u,          5u,          7u,
                                              10u,         63u,         64u,         65u,         255u,
                                              256u,        1000u,       65535u,      65536u,      65537u,
                                              0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu, 0xFFFFFFFFu};
  return kDivs;
}

const std::vector<uint64_t>& divisors64() {
  static const std::vector<uint64_t> kDivs = {
      1ull,          2ull,          3ull,          7ull,          64ull,          1000ull,
      0xFFFFFFFFull, 0x100000000ull, 0x100000001ull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
      0x8000000000000001ull, 0xFFFFFFFFFFFFFFFEull, 0xFFFFFFFFFFFFFFFFull};
  return kDivs;
}

// Dividends around 0, around y and its multiples, and at the top of the range.
template <typename U>
std::vector<U> dividendsFor(U y) {
  const U kMax = static_cast<U>(~U(0));
  std::vector<U> xs = {U(0), U(1), U(2), kMax, U(kMax - 1), U(kMax / 2), U(kMax / 2 + 1)};
  for (U k : {U(1), U(2), U(3), U(kMax / y)}) {
    if (k <= kMax / y) {
      const U m = static_cast<U>(y * k);  // k <= kMax / y, so y * k cannot wrap.
      xs.push_back(m);
      xs.push_back(static_cast<U>(m - 1));
      if (m != kMax) {
        xs.push_back(static_cast<U>(m + 1));
      }
    }
  }
  return xs;
}

void expectDivMod32(uint32_t x, uint32_t y) {
  const uint32_t rcp = nu::idivRcp32(y);
  uint32_t q = 0xDEADBEEFu;
  uint32_t r = 0xDEADBEEFu;
  nu::idivmodFast32(&q, &r, x, y, rcp);
  EXPECT_EQ(q, x / y) << "x=" << x << " y=" << y;
  EXPECT_EQ(r, x % y) << "x=" << x << " y=" << y;
  EXPECT_EQ(nu::idivFast32(x, y, rcp), x / y) << "x=" << x << " y=" << y;
  EXPECT_EQ(nu::imodFast32(x, y, rcp), x % y) << "x=" << x << " y=" << y;
}

void expectDivMod64(uint64_t x, uint64_t y) {
  const uint64_t rcp = nu::idivRcp64(y);
  uint64_t q = 0xDEADBEEFull;
  uint64_t r = 0xDEADBEEFull;
  nu::idivmodFast64(&q, &r, x, y, rcp);
  EXPECT_EQ(q, x / y) << "x=" << x << " y=" << y;
  EXPECT_EQ(r, x % y) << "x=" << x << " y=" << y;
  // idivFast64 returns uint32_t, so it can only ever match the low word of the quotient.
  EXPECT_EQ(nu::idivFast64(x, y, rcp), static_cast<uint32_t>(x / y)) << "x=" << x << " y=" << y;
  EXPECT_EQ(nu::imodFast64(x, y, rcp), x % y) << "x=" << x << " y=" << y;
}

// ---------------------------------------------------------------------------
// Integer divUp / roundUp / roundDown / alignUp / alignDown
// ---------------------------------------------------------------------------

static_assert(nu::divUp(0, 8) == 0 && nu::divUp(1, 8) == 1 && nu::divUp(8, 8) == 1 && nu::divUp(9, 8) == 2,
              "divUp must be usable in constant expressions");
static_assert(nu::roundUp(9, 8) == 16 && nu::roundDown(15, 8) == 8, "roundUp/roundDown constexpr");
static_assert(nu::alignUp(9, 8) == 16 && nu::alignDown(15, 8) == 8, "alignUp/alignDown constexpr");
// Z defaults to decltype(X() + Y()) for divUp..alignUp but decltype(X() + int()) for alignDown.
static_assert(std::is_same<decltype(nu::divUp(uint8_t{1}, uint8_t{1})), int>::value, "promotes like X+Y");
static_assert(std::is_same<decltype(nu::alignUp(uint64_t{1}, 8)), uint64_t>::value, "alignUp Z=X+Y");
static_assert(std::is_same<decltype(nu::alignDown(uint8_t{1}, uint64_t{8})), int>::value,
              "alignDown ignores Y when picking Z");
static_assert(std::is_same<decltype(nu::alignDown(uint64_t{1}, 8)), uint64_t>::value, "alignDown Z=X+int");

TEST(NcclUtilityIntRound, MatchesDivisionReference) {
  for (uint64_t y = 1; y <= 33; ++y) {
    for (uint64_t x = 0; x <= 300; ++x) {
      SCOPED_TRACE(::testing::Message() << "x=" << x << " y=" << y);
      // Smallest q with q * y >= x, found by counting, so no reference shares divUp's (x + y - 1) / y.
      uint64_t q = 0;
      while (q * y < x) {
        ++q;
      }
      const uint64_t down = (x == q * y) ? x : (q - 1) * y;
      EXPECT_EQ(nu::divUp(x, y), q);
      EXPECT_EQ(nu::roundUp(x, y), q * y);
      EXPECT_EQ(nu::roundDown(x, y), down);
      EXPECT_EQ(nu::divUp(static_cast<uint32_t>(x), static_cast<int>(y)), static_cast<uint32_t>(q));
      EXPECT_EQ(nu::roundDown(static_cast<int>(x), static_cast<int>(y)), static_cast<int>(down));
    }
  }
}

TEST(NcclUtilityIntRound, AlignPowerOfTwoMatchesReference) {
  for (uint64_t a = 1; a <= (uint64_t{1} << 12); a <<= 1) {
    for (uint64_t x = 0; x <= 600; ++x) {
      SCOPED_TRACE(::testing::Message() << "x=" << x << " a=" << a);
      EXPECT_EQ(nu::alignUp(x, a), ((x + a - 1) / a) * a);
      EXPECT_EQ(nu::alignDown(x, a), (x / a) * a);
      EXPECT_EQ(nu::alignUp(static_cast<uint32_t>(x), static_cast<int>(a)),
                static_cast<uint32_t>(((x + a - 1) / a) * a));
      EXPECT_EQ(nu::alignDown(static_cast<int>(x), static_cast<int>(a)), static_cast<int>((x / a) * a));
    }
  }
  // The mask is built in Z, so a 64-bit value keeps its high bits against an int alignment.
  EXPECT_EQ(nu::alignDown(uint64_t{0xFFFFFFFFFFFFFFFFull}, 4096), uint64_t{0xFFFFFFFFFFFFF000ull});
  EXPECT_EQ(nu::alignUp(uint64_t{0x100000001ull}, 4096), uint64_t{0x100001000ull});
  EXPECT_EQ(nu::alignDown(size_t{0x123456789ull}, size_t{1} << 32), size_t{0x100000000ull});
}

// Negative inputs: alignDown floors, roundDown truncates toward zero; callers must not swap them for signed values.
TEST(NcclUtilityIntRound, SignedNegativeInputsPinRoundingDirection) {
  EXPECT_EQ(nu::alignDown(-5, 4), -8);
  EXPECT_EQ(nu::roundDown(-5, 4), -4);
  EXPECT_EQ(nu::alignUp(-5, 4), -4);
  EXPECT_EQ(nu::roundUp(-5, 4), 0);
  EXPECT_EQ(nu::divUp(-5, 4), 0);
  EXPECT_EQ(nu::alignDown(-8, 8), -8);
  EXPECT_EQ(nu::alignUp(-8, 8), -8);
}

// No overflow guard: alignUp/roundUp of a value within a-1 of the unsigned max wraps to 0.
TEST(NcclUtilityIntRound, UnsignedOverflowWrapsToZero) {
  EXPECT_EQ(nu::alignUp(uint32_t{0xFFFFFFFFu}, 8u), 0u);
  EXPECT_EQ(nu::roundUp(uint32_t{0xFFFFFFF9u}, 8u), 0u);
  EXPECT_EQ(nu::alignUp(uint32_t{0xFFFFFFF8u}, 8u), 0xFFFFFFF8u);
}

// ---------------------------------------------------------------------------
// Pointer alignUp / alignDown
// ---------------------------------------------------------------------------

template <typename P, typename = void>
struct AlignDownResult {
  using Type = void;
};
template <typename P>
struct AlignDownResult<P, decltype(void(nu::alignDown(std::declval<P>(), size_t{})))> {
  using Type = decltype(nu::alignDown(std::declval<P>(), size_t{}));
};
template <typename P, typename = void>
struct AlignUpResult {
  using Type = void;
};
template <typename P>
struct AlignUpResult<P, decltype(void(nu::alignUp(std::declval<P>(), size_t{})))> {
  using Type = decltype(nu::alignUp(std::declval<P>(), size_t{}));
};

// The byte-pointer overloads preserve the pointee type, including const.
static_assert(std::is_same<AlignDownResult<char*>::Type, char*>::value, "alignDown(T*) returns T*");
static_assert(std::is_same<AlignDownResult<char const*>::Type, char const*>::value, "keeps const");
static_assert(std::is_same<AlignDownResult<uint8_t*>::Type, uint8_t*>::value, "uint8_t deduces T*");
static_assert(std::is_same<AlignUpResult<char const*>::Type, char const*>::value, "alignUp keeps const");
// void const* overloads have a non-deducible T: a plain call picks T* with T = const void, so pass alignDown<AnyT>.
static_assert(std::is_same<AlignDownResult<void const*>::Type, void const*>::value,
              "plain call resolves to T* with T = const void, not to the void const* overload");
static_assert(std::is_same<AlignUpResult<void const*>::Type, void const*>::value,
              "plain call resolves to T* with T = const void, not to the void const* overload");
static_assert(std::is_same<decltype(nu::alignDown<char>(std::declval<void const*>(), size_t{})), void*>::value,
              "explicit-T void const* overload returns non-const void*");
static_assert(std::is_same<decltype(nu::alignUp<char>(std::declval<void const*>(), size_t{})), void*>::value,
              "explicit-T void const* overload returns non-const void*");

class NcclUtilityAlignPtr : public ::testing::Test {
 protected:
  static constexpr size_t kAlignments[] = {1, 2, 4, 8, 16, 32, 64};
  static constexpr size_t kOffsets = 128;
  // 64-byte aligned base, so (base + off) % a == off % a for every a above; 64 spare bytes for alignUp.
  alignas(64) char buf_[kOffsets + 64] = {};
};

TEST_F(NcclUtilityAlignPtr, ByteOverloadsMatchModuloReference) {
  char* const base = buf_;
  ASSERT_EQ(reinterpret_cast<uintptr_t>(base) % 64, 0u);
  for (size_t a : kAlignments) {
    for (size_t off = 0; off < kOffsets; ++off) {
      SCOPED_TRACE(::testing::Message() << "a=" << a << " off=" << off);
      char* const p = base + off;
      char* const down = base + (off - off % a);
      char* const up = base + ((off + a - 1) / a) * a;
      EXPECT_EQ(nu::alignDown(p, a), down);
      EXPECT_EQ(nu::alignUp(p, a), up);
      EXPECT_EQ(nu::alignDown(static_cast<char const*>(p), a), down);
      EXPECT_EQ(nu::alignUp(static_cast<char const*>(p), a), up);
      EXPECT_EQ(nu::alignDown(reinterpret_cast<uint8_t*>(p), a), reinterpret_cast<uint8_t*>(down));
      EXPECT_EQ(nu::alignUp(reinterpret_cast<uint8_t*>(p), a), reinterpret_cast<uint8_t*>(up));
      EXPECT_EQ(nu::alignDown(reinterpret_cast<signed char*>(p), a), reinterpret_cast<signed char*>(down));
      EXPECT_EQ(nu::alignUp(reinterpret_cast<signed char*>(p), a), reinterpret_cast<signed char*>(up));
    }
  }
  EXPECT_EQ(nu::alignDown(base + 9, size_t{8}), base + 8);
  EXPECT_EQ(nu::alignUp(base + 9, size_t{8}), base + 16);
  EXPECT_EQ(nu::alignUp(base + 8, size_t{8}), base + 8);
}

TEST_F(NcclUtilityAlignPtr, VoidConstOverloadsNeedExplicitTemplateArg) {
  char* const base = buf_;
  for (size_t a : kAlignments) {
    for (size_t off = 0; off < kOffsets; ++off) {
      SCOPED_TRACE(::testing::Message() << "a=" << a << " off=" << off);
      void const* const p = base + off;
      void* const down = base + (off - off % a);
      void* const up = base + ((off + a - 1) / a) * a;
      EXPECT_EQ(nu::alignDown<char>(p, a), down);
      EXPECT_EQ(nu::alignUp<char>(p, a), up);
      // T is never used, so any type (even a non-byte one) gives the same address.
      EXPECT_EQ(nu::alignDown<double>(p, a), down);
      EXPECT_EQ(nu::alignUp<double>(p, a), up);
    }
  }
}

// ---------------------------------------------------------------------------
// isPow2 / rollingLessEq / rollingLessThan
// ---------------------------------------------------------------------------

static_assert(nu::isPow2(1u) && nu::isPow2(2u) && nu::isPow2(uint64_t{1} << 63) && !nu::isPow2(3u),
              "isPow2 is constexpr");
static_assert(nu::isPow2(0u), "isPow2(0) is true: the bit trick has no zero guard");

TEST(NcclUtilityPow2, MatchesPopcountExhaustive16Bit) {
  for (uint32_t x = 0; x <= 0xFFFFu; ++x) {
    ASSERT_EQ(nu::isPow2(x), __builtin_popcount(x) <= 1) << "x=" << x;
    ASSERT_EQ(nu::isPow2(static_cast<uint16_t>(x)), __builtin_popcount(x) <= 1) << "x=" << x;
  }
  for (int s = 0; s < 64; ++s) {
    const uint64_t p = uint64_t{1} << s;
    EXPECT_TRUE(nu::isPow2(p)) << "s=" << s;
    if (s > 0) {
      EXPECT_FALSE(nu::isPow2(p + 1)) << "s=" << s;
    }
    if (s > 1) {
      EXPECT_FALSE(nu::isPow2(p - 1)) << "s=" << s;
    }
  }
  EXPECT_TRUE(nu::isPow2(0));
  EXPECT_FALSE(nu::isPow2(0xFFFFFFFFu));
}

TEST(NcclUtilityRolling, FullWidthWrapsAtHalfRange) {
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(5, 5));
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(5, 6));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(6, 5));
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0xFFFFFFFFu, 0));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(0, 0xFFFFFFFFu));
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0, 0x7FFFFFFFu));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(0, 0x80000000u));
  EXPECT_FALSE(nu::rollingLessThan<uint32_t>(5, 5));
  EXPECT_TRUE(nu::rollingLessThan<uint32_t>(5, 6));
  EXPECT_FALSE(nu::rollingLessThan<uint32_t>(6, 5));
  EXPECT_TRUE(nu::rollingLessThan<uint32_t>(0xFFFFFFFFu, 0));
  EXPECT_TRUE(nu::rollingLessEq<uint64_t>(~uint64_t{0}, 1));
  EXPECT_FALSE(nu::rollingLessEq<uint64_t>(0, uint64_t{1} << 63));
  EXPECT_TRUE(nu::rollingLessEq<uint64_t>(0, (uint64_t{1} << 63) - 1));
  EXPECT_TRUE(nu::rollingLessThan<uint64_t>(~uint64_t{0}, 0));
}

// Exhaustive over all uint8 pairs, as uint8 and as a wider type truncated to nBits = 8 with garbage high bits.
TEST(NcclUtilityRolling, EightBitWindowExhaustive) {
  for (uint32_t a = 0; a < 256; ++a) {
    for (uint32_t b = 0; b < 256; ++b) {
      const bool lessEq = ((b - a) & 0xFFu) <= 0x7Fu;
      const bool lessThan = ((a - b) & 0xFFu) > 0x7Fu;
      ASSERT_EQ(nu::rollingLessEq<uint8_t>(a, b), lessEq) << a << " " << b;
      ASSERT_EQ(nu::rollingLessThan<uint8_t>(a, b), lessThan) << a << " " << b;
      const uint32_t ha = a | 0xABCD0000u;
      const uint32_t hb = b | 0x12340000u;
      ASSERT_EQ(nu::rollingLessEq<uint32_t>(ha, hb, 8), lessEq) << a << " " << b;
      ASSERT_EQ(nu::rollingLessThan<uint32_t>(ha, hb, 8), lessThan) << a << " " << b;
    }
  }
}

// gin_barrier__funcs.h compares 64-bit signals in a 32-bit window: only the low 32 bits may matter.
TEST(NcclUtilityRolling, SubWidthWindowIgnoresHighBits) {
  EXPECT_TRUE(nu::rollingLessEq<uint64_t>(0xFFFFFFFFull, 0x1'00000000ull, 32));
  EXPECT_TRUE(nu::rollingLessEq<uint64_t>(0x5'FFFFFFF0ull, 0x2'00000010ull, 32));
  EXPECT_FALSE(nu::rollingLessEq<uint64_t>(0x00000010ull, 0x7'FFFFFFF0ull, 32));
  EXPECT_TRUE(nu::rollingLessEq<uint64_t>(0, 0x7FFFFFFFull, 32));
  EXPECT_FALSE(nu::rollingLessEq<uint64_t>(0, 0x80000000ull, 32));
  EXPECT_TRUE(nu::rollingLessThan<uint64_t>(0x9'FFFFFFFFull, 0x1'00000000ull, 32));
  EXPECT_FALSE(nu::rollingLessThan<uint64_t>(0x9'00000007ull, 0x1'00000007ull, 32));
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0xFFF, 0x001, 12));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(0x001, 0xFFF, 12));
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0xFFF, 0x000, 12));
  // A 1-bit window has a half-range of 0, so only equal values compare less-or-equal.
  EXPECT_TRUE(nu::rollingLessEq<uint32_t>(0, 0, 1));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(1, 0, 1));
  EXPECT_FALSE(nu::rollingLessEq<uint32_t>(0, 1, 1));
}

// ---------------------------------------------------------------------------
// Reciprocal division
// ---------------------------------------------------------------------------

static_assert(nu::idivRcp32(1) == 0, "y=1: UINT32_MAX/1 + 1 wraps, and idivmodFast32 treats rcp 0 as divide-by-1");
static_assert(nu::idivRcp32(2) == 0x80000000u && nu::idivRcp32(3) == 0x55555555u, "idivRcp32 constexpr");
static_assert(nu::idivRcp32(0x80000000u) == 2 && nu::idivRcp32(0xFFFFFFFFu) == 1, "idivRcp32 top of range");
static_assert(nu::idivRcp64(1) == 0 && nu::idivRcp64(2) == 0x8000000000000000ull, "idivRcp64 constexpr");
static_assert(nu::idivRcp64(3) == 0x5555555555555555ull && nu::idivRcp64(~uint64_t{0}) == 1, "idivRcp64 values");
static_assert(std::is_same<decltype(nu::idivFast64(0, 1, 0)), uint32_t>::value,
              "idivFast64 narrows its 64-bit quotient to uint32_t");

TEST(NcclUtilityRcp, ReciprocalFormula) {
  for (uint32_t y : divisors32()) {
    expectRcpProperty<32>(y, nu::idivRcp32(y));
  }
  for (uint64_t y : divisors64()) {
    expectRcpProperty<64>(y, nu::idivRcp64(y));
  }
  for (uint32_t y = 1; y <= 5000; ++y) {
    expectRcpProperty<32>(y, nu::idivRcp32(y));
    expectRcpProperty<64>(y, nu::idivRcp64(y));
  }
  XorShift64 rng(0x7E57AB1Eull);
  for (int i = 0; i < 20000; ++i) {
    const uint64_t y = std::max<uint64_t>(1u, rng.next() >> (rng.next() % 64));
    expectRcpProperty<32>(static_cast<uint32_t>(y) | 1u, nu::idivRcp32(static_cast<uint32_t>(y) | 1u));
    expectRcpProperty<64>(y, nu::idivRcp64(y));
  }
  for (int s = 1; s < 64; ++s) {
    EXPECT_EQ(nu::idivRcp64(uint64_t{1} << s), uint64_t{1} << (64 - s)) << "s=" << s;
  }
  for (int s = 1; s < 32; ++s) {
    EXPECT_EQ(nu::idivRcp32(1u << s), 1u << (32 - s)) << "s=" << s;
  }
}

TEST(NcclUtilityRcp, MulHiMatchesWideReference) {
  EXPECT_EQ(nu::mul32hi(0xFFFFFFFFu, 0xFFFFFFFFu), 0xFFFFFFFEu);
  EXPECT_EQ(nu::mul32hi(0x80000000u, 2u), 1u);
  EXPECT_EQ(nu::mul32hi(0x7FFFFFFFu, 1u), 0u);
  EXPECT_EQ(nu::mul64hi(~uint64_t{0}, ~uint64_t{0}), ~uint64_t{0} - 1);
  EXPECT_EQ(nu::mul64hi(uint64_t{1} << 63, 2), 1u);
  EXPECT_EQ(nu::mul64hi(0x123456789ABCDEFull, 0xFEDCBA987654321ull), 0x121FA00AD77D74ull);
  XorShift64 rng(0x9E3779B97F4A7C15ull);
  for (int i = 0; i < 200000; ++i) {
    const uint64_t a = rng.next();
    const uint64_t b = rng.next();
    ASSERT_EQ(nu::mul32hi(static_cast<uint32_t>(a), static_cast<uint32_t>(b)),
              refMul32hi(static_cast<uint32_t>(a), static_cast<uint32_t>(b)))
        << a << " " << b;
    ASSERT_EQ(nu::mul64hi(a, b), refMul64hi(a, b)) << a << " " << b;
  }
}

TEST(NcclUtilityRcp, DivModFast32EdgesAndSweep) {
  for (uint32_t y : divisors32()) {
    for (uint32_t x : dividendsFor<uint32_t>(y)) {
      expectDivMod32(x, y);
    }
  }
  // Exhaustive low range for small divisors, where the single +1 correction fires most often.
  for (uint32_t y = 1; y <= 300; ++y) {
    const uint32_t rcp = nu::idivRcp32(y);
    for (uint32_t x = 0; x < (1u << 14); ++x) {
      uint32_t q = 0;
      uint32_t r = 0;
      nu::idivmodFast32(&q, &r, x, y, rcp);
      ASSERT_EQ(q, x / y) << "x=" << x << " y=" << y;
      ASSERT_EQ(r, x % y) << "x=" << x << " y=" << y;
    }
    for (uint32_t x = 0xFFFFFFFFu - (1u << 12); x != 0; ++x) {
      uint32_t q = 0;
      uint32_t r = 0;
      nu::idivmodFast32(&q, &r, x, y, rcp);
      ASSERT_EQ(q, x / y) << "x=" << x << " y=" << y;
      ASSERT_EQ(r, x % y) << "x=" << x << " y=" << y;
    }
  }
  XorShift64 rng(0xC0FFEE1234ull);
  for (int i = 0; i < 500000; ++i) {
    const uint32_t y = std::max<uint32_t>(1u, static_cast<uint32_t>(rng.next()) >> (rng.next() % 32));
    const uint32_t x = static_cast<uint32_t>(rng.next());
    uint32_t q = 0;
    uint32_t r = 0;
    nu::idivmodFast32(&q, &r, x, y, nu::idivRcp32(y));
    ASSERT_EQ(q, x / y) << "x=" << x << " y=" << y;
    ASSERT_EQ(r, x % y) << "x=" << x << " y=" << y;
  }
}

TEST(NcclUtilityRcp, DivModFast64EdgesAndSweep) {
  for (uint64_t y : divisors64()) {
    for (uint64_t x : dividendsFor<uint64_t>(y)) {
      expectDivMod64(x, y);
    }
  }
  XorShift64 rng(0xBADC0DE5EEDull);
  for (int i = 0; i < 500000; ++i) {
    const uint64_t y = std::max<uint64_t>(1u, rng.next() >> (rng.next() % 64));
    const uint64_t x = rng.next();
    uint64_t q = 0;
    uint64_t r = 0;
    nu::idivmodFast64(&q, &r, x, y, nu::idivRcp64(y));
    ASSERT_EQ(q, x / y) << "x=" << x << " y=" << y;
    ASSERT_EQ(r, x % y) << "x=" << x << " y=" << y;
  }
}

// idivFast64 returns uint32_t, so a quotient >= 2^32 silently loses its high word; imodFast64 is unaffected.
TEST(NcclUtilityRcp, IdivFast64TruncatesWideQuotient) {
  const uint64_t x = (uint64_t{1} << 40) + 5;
  EXPECT_EQ(nu::idivFast64(x, 1, nu::idivRcp64(1)), 5u)
      << "idivFast64 no longer truncates the quotient: delete this pin and uncomment the check below";
  // EXPECT_EQ(nu::idivFast64(x, 1, nu::idivRcp64(1)), x);  // Correct if idivFast64 returned uint64_t.
  EXPECT_EQ(nu::idivFast64(x, 4, nu::idivRcp64(4)), static_cast<uint32_t>(x / 4));
  EXPECT_EQ(nu::imodFast64(x, 4, nu::idivRcp64(4)), x % 4);
  uint64_t q = 0;
  uint64_t r = 0;
  nu::idivmodFast64(&q, &r, x, 1, nu::idivRcp64(1));
  EXPECT_EQ(q, x);
  EXPECT_EQ(r, 0u);
}

// A composed reciprocal must equal idivRcp of the product. The one known exception is checked by its own test below.
bool isKnownImulRcpZeroCase(uint64_t got, uint64_t want) {
  return got == 0 && want == 1;
}

// imulRcp(x, rcp(x), y, rcp(y)) equals idivRcp(x*y), so it divides by x*y exactly, except when it returns 0 for 1.
TEST(NcclUtilityRcp, ImulRcpComposesReciprocals) {
  // rcp 0 (factor 1) shortcuts to the other reciprocal.
  EXPECT_EQ(nu::imulRcp32(1, nu::idivRcp32(1), 7, nu::idivRcp32(7)), nu::idivRcp32(7));
  EXPECT_EQ(nu::imulRcp32(7, nu::idivRcp32(7), 1, nu::idivRcp32(1)), nu::idivRcp32(7));
  EXPECT_EQ(nu::imulRcp32(1, 0, 1, 0), 0u);
  EXPECT_EQ(nu::imulRcp64(1, nu::idivRcp64(1), 7, nu::idivRcp64(7)), nu::idivRcp64(7));
  EXPECT_EQ(nu::imulRcp64(7, nu::idivRcp64(7), 1, nu::idivRcp64(1)), nu::idivRcp64(7));
  // Products that need the +1 correction (pow2) and products that do not.
  EXPECT_EQ(nu::imulRcp32(2, nu::idivRcp32(2), 2, nu::idivRcp32(2)), nu::idivRcp32(4));
  EXPECT_EQ(nu::imulRcp32(3, nu::idivRcp32(3), 5, nu::idivRcp32(5)), nu::idivRcp32(15));
  EXPECT_EQ(nu::imulRcp64(3, nu::idivRcp64(3), 5, nu::idivRcp64(5)), nu::idivRcp64(15));
  for (uint32_t x = 1; x <= 200; ++x) {
    for (uint32_t y = 1; y <= 200; ++y) {
      const uint32_t d = x * y;
      const uint32_t rcp = nu::imulRcp32(x, nu::idivRcp32(x), y, nu::idivRcp32(y));
      ASSERT_EQ(rcp, nu::idivRcp32(d)) << "x=" << x << " y=" << y;
      const uint64_t wide = uint64_t{x} * y;
      ASSERT_EQ(nu::imulRcp64(x, nu::idivRcp64(x), y, nu::idivRcp64(y)), nu::idivRcp64(wide))
          << "x=" << x << " y=" << y;
      for (uint32_t n : {0u, 1u, d - 1, d, d + 1, 0x7FFFFFFFu, 0xFFFFFFFFu}) {
        ASSERT_EQ(nu::idivFast32(n, d, rcp), n / d) << "n=" << n << " d=" << d;
      }
    }
  }
  XorShift64 rng(0x5151515151ull);
  int zeroCases32 = 0;
  int zeroCases64 = 0;
  for (int i = 0; i < 400000; ++i) {
    const uint32_t x = std::max<uint32_t>(1u, static_cast<uint32_t>(rng.next()) >> (rng.next() % 32));
    const uint32_t y = static_cast<uint32_t>(rng.next() % (0xFFFFFFFFu / x)) + 1;
    const uint32_t d = x * y;
    const uint32_t rcp = nu::imulRcp32(x, nu::idivRcp32(x), y, nu::idivRcp32(y));
    if (isKnownImulRcpZeroCase(rcp, nu::idivRcp32(d))) {
      ++zeroCases32;
    } else {
      ASSERT_EQ(rcp, nu::idivRcp32(d)) << "x=" << x << " y=" << y;
      const uint32_t n = static_cast<uint32_t>(rng.next());
      ASSERT_EQ(nu::idivFast32(n, d, rcp), n / d) << "n=" << n << " d=" << d;
    }
    const uint64_t bigX = std::max<uint64_t>(1u, rng.next() >> (rng.next() % 64));
    const uint64_t bigY = rng.next() % (~uint64_t{0} / bigX) + 1;
    const uint64_t bigD = bigX * bigY;
    const uint64_t rcp64 = nu::imulRcp64(bigX, nu::idivRcp64(bigX), bigY, nu::idivRcp64(bigY));
    if (isKnownImulRcpZeroCase(rcp64, nu::idivRcp64(bigD))) {
      ++zeroCases64;
    } else {
      ASSERT_EQ(rcp64, nu::idivRcp64(bigD)) << "X=" << bigX << " Y=" << bigY;
      const uint64_t n = rng.next();
      ASSERT_EQ(nu::imodFast64(n, bigD, rcp64), n % bigD) << "n=" << n << " D=" << bigD;
    }
  }
  // The sweep reaches the zero case often; if it never shows up, the generator stopped covering large products.
  EXPECT_GT(zeroCases32, 0) << "imulRcp32 no longer returns 0 when the true reciprocal is 1: delete this counter";
  EXPECT_GT(zeroCases64, 0) << "imulRcp64 no longer returns 0 when the true reciprocal is 1: delete this counter";
}

// When the true reciprocal is 1, imulRcp returns 0 (the divide-by-1 sentinel), so idivFast returns x unchanged.
TEST(NcclUtilityRcp, ImulRcpReturnsZeroWhenTrueReciprocalIsOne) {
  const uint32_t x = 34;
  const uint32_t y = 123489041;
  const uint32_t d = x * y;  // 4198627394, so the true reciprocal is 1.
  ASSERT_EQ(nu::idivRcp32(d), 1u);
  const uint32_t rcp = nu::imulRcp32(x, nu::idivRcp32(x), y, nu::idivRcp32(y));
  EXPECT_EQ(rcp, 0u) << "imulRcp32 now returns the true reciprocal: delete this pin and uncomment the check below";
  // EXPECT_EQ(rcp, 1u);  // Correct with `if (rcp == 0 || x * y <= rem) rcp += 1;` in imulRcp32.
  EXPECT_EQ(nu::idivFast32(0xFFFFFFFFu, d, rcp), 0xFFFFFFFFu)
      << "imulRcp32 now returns the true reciprocal: delete this pin and uncomment the check below";
  // EXPECT_EQ(nu::idivFast32(0xFFFFFFFFu, d, rcp), 1u);  // Correct once imulRcp32 returns 1 here.
  const uint64_t bigX = 3;
  const uint64_t bigY = 5560062487026735461ull;
  const uint64_t bigD = bigX * bigY;
  ASSERT_EQ(nu::idivRcp64(bigD), 1u);
  const uint64_t rcp64 = nu::imulRcp64(bigX, nu::idivRcp64(bigX), bigY, nu::idivRcp64(bigY));
  EXPECT_EQ(rcp64, 0u) << "imulRcp64 now returns the true reciprocal: delete this pin and uncomment the check below";
  // EXPECT_EQ(rcp64, 1u);  // Correct with `if (rcp == 0 || x * y <= rem) rcp += 1;` in imulRcp64.
  EXPECT_EQ(nu::imodFast64(~uint64_t{0}, bigD, rcp64), bigD - 1)
      << "imulRcp64 now returns the true reciprocal: delete this pin and uncomment the check below";
  // EXPECT_EQ(nu::imodFast64(~uint64_t{0}, bigD, rcp64), ~uint64_t{0} % bigD);  // Correct once imulRcp64 returns 1 here.
}

// ---------------------------------------------------------------------------
// Memory-order mapping tables
// ---------------------------------------------------------------------------

static_assert(nu::acquireOrderOf(std::memory_order_relaxed) == std::memory_order_relaxed, "");
static_assert(nu::acquireOrderOf(std::memory_order_acquire) == std::memory_order_acquire, "");
static_assert(nu::acquireOrderOf(std::memory_order_release) == std::memory_order_relaxed, "");
static_assert(nu::acquireOrderOf(std::memory_order_acq_rel) == std::memory_order_acquire, "");
static_assert(nu::acquireOrderOf(std::memory_order_seq_cst) == std::memory_order_seq_cst, "");
static_assert(nu::releaseOrderOf(std::memory_order_relaxed) == std::memory_order_relaxed, "");
static_assert(nu::releaseOrderOf(std::memory_order_acquire) == std::memory_order_relaxed, "");
static_assert(nu::releaseOrderOf(std::memory_order_release) == std::memory_order_release, "");
static_assert(nu::releaseOrderOf(std::memory_order_acq_rel) == std::memory_order_release, "");
static_assert(nu::releaseOrderOf(std::memory_order_seq_cst) == std::memory_order_seq_cst, "");
static_assert(nu::acquireOrderOf(cuda::memory_order_release) == cuda::memory_order_relaxed, "");
static_assert(nu::acquireOrderOf(cuda::memory_order_acq_rel) == cuda::memory_order_acquire, "");
static_assert(nu::acquireOrderOf(cuda::memory_order_seq_cst) == cuda::memory_order_seq_cst, "");
static_assert(nu::releaseOrderOf(cuda::memory_order_acquire) == cuda::memory_order_relaxed, "");
static_assert(nu::releaseOrderOf(cuda::memory_order_acq_rel) == cuda::memory_order_release, "");
static_assert(nu::releaseOrderOf(cuda::memory_order_seq_cst) == cuda::memory_order_seq_cst, "");
static_assert(nu::toAtomicBuiltinOrder(std::memory_order_relaxed) == __ATOMIC_RELAXED, "");
static_assert(nu::toAtomicBuiltinOrder(std::memory_order_acq_rel) == __ATOMIC_ACQ_REL, "");
static_assert(nu::toCudaOrder(std::memory_order_release) == cuda::memory_order_release, "");
static_assert(nu::toCudaOrder(std::memory_order_seq_cst) == cuda::memory_order_seq_cst, "");

struct OrderRow {
  std::memory_order stdOrder;
  cuda::memory_order cudaOrder;
  std::memory_order acquire;
  std::memory_order release;
  int builtin;
  cuda::memory_order cudaOfStd;
};

// consume has no cuda:: twin: acquire/releaseOrderOf pass it through, the builtin and cuda maps fall back to seq_cst.
TEST(NcclUtilityOrder, MappingTablesAtRuntime) {
  const std::array<OrderRow, 6> rows = {{
      {std::memory_order_relaxed, cuda::memory_order_relaxed, std::memory_order_relaxed, std::memory_order_relaxed,
       __ATOMIC_RELAXED, cuda::memory_order_relaxed},
      {std::memory_order_acquire, cuda::memory_order_acquire, std::memory_order_acquire, std::memory_order_relaxed,
       __ATOMIC_ACQUIRE, cuda::memory_order_acquire},
      {std::memory_order_release, cuda::memory_order_release, std::memory_order_relaxed, std::memory_order_release,
       __ATOMIC_RELEASE, cuda::memory_order_release},
      {std::memory_order_acq_rel, cuda::memory_order_acq_rel, std::memory_order_acquire, std::memory_order_release,
       __ATOMIC_ACQ_REL, cuda::memory_order_acq_rel},
      {std::memory_order_seq_cst, cuda::memory_order_seq_cst, std::memory_order_seq_cst, std::memory_order_seq_cst,
       __ATOMIC_SEQ_CST, cuda::memory_order_seq_cst},
      {std::memory_order_consume, cuda::memory_order_seq_cst, std::memory_order_consume, std::memory_order_consume,
       __ATOMIC_SEQ_CST, cuda::memory_order_seq_cst},
  }};
  for (const OrderRow& row : rows) {
    SCOPED_TRACE(::testing::Message() << "std order " << static_cast<int>(row.stdOrder));
    EXPECT_EQ(nu::acquireOrderOf(row.stdOrder), row.acquire);
    EXPECT_EQ(nu::releaseOrderOf(row.stdOrder), row.release);
    EXPECT_EQ(nu::toAtomicBuiltinOrder(row.stdOrder), row.builtin);
    EXPECT_EQ(nu::toCudaOrder(row.stdOrder), row.cudaOfStd);
    if (row.stdOrder != std::memory_order_consume) {
      EXPECT_EQ(nu::acquireOrderOf(row.cudaOrder), nu::toCudaOrder(row.acquire));
      EXPECT_EQ(nu::releaseOrderOf(row.cudaOrder), nu::toCudaOrder(row.release));
    }
  }
}

// ---------------------------------------------------------------------------
// Type utilities: declval, always_false, ValueAsType, IntSeq, Present, Absent, Optional
// ---------------------------------------------------------------------------

static_assert(std::is_same<decltype(nu::declval<int>()), int&&>::value, "declval<T>() is T&&");
static_assert(std::is_same<decltype(nu::declval<int&>()), int&>::value, "declval<T&>() collapses to T&");
static_assert(std::is_same<decltype(nu::declval<std::vector<int>>().size()), size_t>::value,
              "declval works in unevaluated member-call contexts");
static_assert(!nu::always_false<int>::value && !nu::always_false<void>::value, "always_false is false");
static_assert(nu::ValueAsType<int, 42>::value == 42, "ValueAsType carries the value");
static_assert(nu::ValueAsType<uint64_t, ~uint64_t{0}>::value == ~uint64_t{0}, "full-width value");
static_assert(std::is_same<decltype(nu::ValueAsType<char, 'x'>::value), const char>::value, "value has type T");
static_assert(std::is_same<nu::IntSeqUpTo<0, 0>::Type, nu::IntSeq<>>::value, "empty sequence");
static_assert(std::is_same<nu::IntSeqUpTo<1, 0>::Type, nu::IntSeq<0>>::value, "one element");
static_assert(std::is_same<nu::IntSeqUpTo<4, 0>::Type, nu::IntSeq<0, 1, 2, 3>>::value, "0..n-1");
static_assert(std::is_same<nu::IntSeqUpTo<4, 2>::Type, nu::IntSeq<2, 3>>::value, "starts at m");
static_assert(std::is_same<decltype(nu::present()), nu::Present<>>::value, "present() is empty");
static_assert(std::is_same<decltype(nu::present(nu::declval<int&>(), 1)), nu::Present<int&, int&&>>::value,
              "present() keeps value categories as references");

// always_false lets a discarded if-constexpr branch hold a static_assert that only fires if instantiated.
template <typename T>
int onlyIntegral(T v) {
  if constexpr (std::is_integral<T>::value) {
    return static_cast<int>(v);
  } else {
    static_assert(nu::always_false<T>::value, "only integral types");
    return 0;
  }
}

TEST(NcclUtilityTypes, AlwaysFalseOnlyFiresInInstantiatedBranch) {
  EXPECT_EQ(onlyIntegral(7), 7);
  EXPECT_EQ(onlyIntegral(uint8_t{9}), 9);
}

struct Tracked {
  static int ctors;
  static int dtors;
  int a;
  int b;
  Tracked(int a_, int b_) : a(a_), b(b_) {
    ++ctors;
  }
  ~Tracked() {
    ++dtors;
  }
};
int Tracked::ctors = 0;
int Tracked::dtors = 0;

struct RefHolder {
  int& ref;
};

struct UniqueHolder {
  explicit UniqueHolder(std::unique_ptr<int> p) : ptr(std::move(p)) {}
  std::unique_ptr<int> ptr;
};

class NcclUtilityOptional : public ::testing::Test {
 protected:
  void SetUp() override {
    Tracked::ctors = 0;
    Tracked::dtors = 0;
  }
};

TEST_F(NcclUtilityOptional, PresentArgsConstructAndDestructOnce) {
  {
    nu::Optional<Tracked> opt(nu::present(3, 4));
    ASSERT_TRUE(opt.present);
    EXPECT_EQ(opt.thing.a, 3);
    EXPECT_EQ(opt.thing.b, 4);
    EXPECT_EQ(Tracked::ctors, 1);
    EXPECT_EQ(Tracked::dtors, 0);
  }
  EXPECT_EQ(Tracked::ctors, 1);
  EXPECT_EQ(Tracked::dtors, 1);
}

TEST_F(NcclUtilityOptional, AbsentAndDefaultNeverConstructOrDestruct) {
  {
    nu::Optional<Tracked> absent(nu::Absent{});
    nu::Optional<Tracked> dflt;
    EXPECT_FALSE(absent.present);
    EXPECT_FALSE(dflt.present);
  }
  EXPECT_EQ(Tracked::ctors, 0);
  EXPECT_EQ(Tracked::dtors, 0);
}

TEST_F(NcclUtilityOptional, ZeroArgPresentValueInitializes) {
  nu::Optional<int> opt(nu::present());
  EXPECT_TRUE(opt.present);
  EXPECT_EQ(opt.thing, 0);
}

TEST_F(NcclUtilityOptional, ForwardsLvaluesAsReferencesAndRvaluesAsMoves) {
  int x = 5;
  {
    nu::Optional<RefHolder> opt(nu::present(x));
    ASSERT_TRUE(opt.present);
    EXPECT_EQ(&opt.thing.ref, &x);
    opt.thing.ref = 7;
  }
  EXPECT_EQ(x, 7);
  // A unique_ptr only compiles here if Present forwards the rvalue all the way to the constructor.
  auto up = std::make_unique<int>(11);
  int* const raw = up.get();
  nu::Optional<UniqueHolder> holder(nu::present(std::move(up)));
  ASSERT_TRUE(holder.present);
  EXPECT_EQ(holder.thing.ptr.get(), raw);
  EXPECT_EQ(up, nullptr);
}

TEST(NcclUtilityTypes, PresentGetAndCopyKeepReferences) {
  int x = 1;
  int y = 2;
  auto p = nu::present(x, y);
  EXPECT_EQ(&p.get(nu::IntSeq<0>{}), &x);
  EXPECT_EQ(&p.get(nu::IntSeq<1>{}), &y);
  auto copy = p;
  EXPECT_EQ(&copy.get(nu::IntSeq<0>{}), &x);
  EXPECT_EQ(&copy.get(nu::IntSeq<1>{}), &y);
  copy.get(nu::IntSeq<1>{}) = 20;
  EXPECT_EQ(y, 20);
}

}  // namespace
}  // namespace RcclUnitTesting
