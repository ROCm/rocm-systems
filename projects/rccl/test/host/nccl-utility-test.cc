/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the pure host/device math helpers in
// src/include/nccl_device/utility.h: rounding/alignment arithmetic, the
// fast-division reciprocal family, the memory_order conversion helpers, and
// the Optional<T>/Present<T...> variadic construction helpers. All targeted
// functions are NCCL_HOST_DEVICE_INLINE with no HIP/device-only dependency,
// so the real header compiles and runs unmodified in a host-only TU -- no
// fakes needed.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>

#include "nccl_device/utility.h"

using nccl::utility::acquireOrderOf;
using nccl::utility::add4G;
using nccl::utility::alignDown;
using nccl::utility::alignUp;
using nccl::utility::divUp;
using nccl::utility::idivFast32;
using nccl::utility::idivFast64;
using nccl::utility::idivmodFast32;
using nccl::utility::idivmodFast64;
using nccl::utility::idivRcp32;
using nccl::utility::idivRcp64;
using nccl::utility::imodFast32;
using nccl::utility::imodFast64;
using nccl::utility::imulRcp32;
using nccl::utility::imulRcp64;
using nccl::utility::isPow2;
using nccl::utility::mul32hi;
using nccl::utility::mul64hi;
using nccl::utility::Optional;
using nccl::utility::present;
using nccl::utility::Present;
using nccl::utility::releaseOrderOf;
using nccl::utility::roundDown;
using nccl::utility::roundUp;
using nccl::utility::rollingLessEq;
using nccl::utility::rollingLessThan;
using nccl::utility::toAtomicBuiltinOrder;
using nccl::utility::toCudaOrder;

namespace {

// Brute-force reference: idivFast32/64 claim idivFast(x, y, idivRcp(y)) == x/y
// for every valid (x, y); these helpers recompute that division the ordinary
// way so the fast path can't share a bug with its own golden.
uint32_t refDiv32(uint32_t x, uint32_t y) { return x / y; }
uint64_t refDiv64(uint64_t x, uint64_t y) { return x / y; }

} // namespace

TEST(NcclUtilityMicrotest, DivUp_RoundsAwayFromZeroOnlyWhenRemainderIsNonzero) {
  EXPECT_EQ(divUp(0u, 4u), 0u);
  EXPECT_EQ(divUp(1u, 4u), 1u);
  EXPECT_EQ(divUp(4u, 4u), 1u);
  EXPECT_EQ(divUp(5u, 4u), 2u);
  EXPECT_EQ(divUp(7u, 3u), 3u);
}

TEST(NcclUtilityMicrotest, RoundUpDown_AreInverseDirectionsAroundAMultiple) {
  EXPECT_EQ(roundUp(0u, 8u), 0u);
  EXPECT_EQ(roundUp(1u, 8u), 8u);
  EXPECT_EQ(roundUp(8u, 8u), 8u);
  EXPECT_EQ(roundUp(9u, 8u), 16u);

  EXPECT_EQ(roundDown(0u, 8u), 0u);
  EXPECT_EQ(roundDown(7u, 8u), 0u);
  EXPECT_EQ(roundDown(8u, 8u), 8u);
  EXPECT_EQ(roundDown(15u, 8u), 8u);
}

TEST(NcclUtilityMicrotest, AlignUpDown_ScalarOverload_AssumesPowerOfTwoAlignment) {
  EXPECT_EQ(alignUp(0u, 16u), 0u);
  EXPECT_EQ(alignUp(1u, 16u), 16u);
  EXPECT_EQ(alignUp(16u, 16u), 16u);
  EXPECT_EQ(alignUp(17u, 16u), 32u);

  EXPECT_EQ(alignDown(0u, 16u), 0u);
  EXPECT_EQ(alignDown(15u, 16u), 0u);
  EXPECT_EQ(alignDown(16u, 16u), 16u);
  EXPECT_EQ(alignDown(31u, 16u), 16u);
}

TEST(NcclUtilityMicrotest, AlignUpDown_PointerOverload_RoundsTheAddressNotTheByte) {
  alignas(32) char buf[64];
  // Single-byte-type overload: static_assert(sizeof(T) == 1) gates this at
  // compile time, so the test itself is part of the contract being checked.
  char* base = buf + 3;
  char* up = alignUp(base, size_t(16));
  char* down = alignDown(base, size_t(16));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(up) % 16, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(down) % 16, 0u);
  EXPECT_GE(up, base);
  EXPECT_LE(down, base);
  EXPECT_LT(reinterpret_cast<uintptr_t>(up) - reinterpret_cast<uintptr_t>(down), 32u);
}

TEST(NcclUtilityMicrotest, IsPow2_TrueOnlyForPowersOfTwoAndZero) {
  EXPECT_TRUE(isPow2(0u));
  EXPECT_TRUE(isPow2(1u));
  EXPECT_TRUE(isPow2(2u));
  EXPECT_TRUE(isPow2(1024u));
  EXPECT_FALSE(isPow2(3u));
  EXPECT_FALSE(isPow2(6u));
  EXPECT_FALSE(isPow2(1023u));
}

TEST(NcclUtilityMicrotest, RollingLessEqAndLessThan_WrapAroundHalfTheRangeCorrectly) {
  // 8-bit rolling counter: 250 is "less than" 4 (wrapped), but not vice versa.
  EXPECT_TRUE(rollingLessEq(uint8_t(250), uint8_t(4), 8));
  EXPECT_FALSE(rollingLessEq(uint8_t(4), uint8_t(250), 8));
  EXPECT_TRUE(rollingLessThan(uint8_t(250), uint8_t(4), 8));
  EXPECT_FALSE(rollingLessThan(uint8_t(4), uint8_t(250), 8));

  // Equal values: LessEq true, LessThan false (strict).
  EXPECT_TRUE(rollingLessEq(uint8_t(7), uint8_t(7), 8));
  EXPECT_FALSE(rollingLessThan(uint8_t(7), uint8_t(7), 8));

  // rollingLessThan is defined as !rollingLessEq(b, a); verify that identity
  // directly so a copy-paste swap of a/b in either function is caught.
  for (uint8_t a : {uint8_t(0), uint8_t(1), uint8_t(127), uint8_t(128), uint8_t(200)}) {
    for (uint8_t b : {uint8_t(0), uint8_t(1), uint8_t(127), uint8_t(128), uint8_t(200)}) {
      EXPECT_EQ(rollingLessThan(a, b, 8), !rollingLessEq(b, a, 8)) << "a=" << int(a) << " b=" << int(b);
    }
  }
}

TEST(NcclUtilityMicrotest, IdivRcp32_MatchesExactFormulaIncludingPow2Correction) {
  // Formula: uint32_t(-1)/x + isPow2(x). The +1 correction for powers of two
  // is what makes idivFast32 exact when y is a power of two; drop it and the
  // round-trip test below would start failing for y in {1,2,4,...}.
  EXPECT_EQ(idivRcp32(1u), uint32_t(-1) / 1u + 1u);
  EXPECT_EQ(idivRcp32(2u), uint32_t(-1) / 2u + 1u);
  EXPECT_EQ(idivRcp32(3u), uint32_t(-1) / 3u + 0u);
  EXPECT_EQ(idivRcp32(100u), uint32_t(-1) / 100u + 0u);
}

TEST(NcclUtilityMicrotest, IdivRcp64_MatchesExactFormulaIncludingPow2Correction) {
  EXPECT_EQ(idivRcp64(1ull), uint64_t(-1) / 1ull + 1ull);
  EXPECT_EQ(idivRcp64(2ull), uint64_t(-1) / 2ull + 1ull);
  EXPECT_EQ(idivRcp64(3ull), uint64_t(-1) / 3ull + 0ull);
  EXPECT_EQ(idivRcp64(1ull << 40), uint64_t(-1) / (1ull << 40) + 1ull);
}

TEST(NcclUtilityMicrotest, Mul32Hi_ReturnsTheHighHalfOfTheFull64BitProduct) {
  EXPECT_EQ(mul32hi(0u, 0xFFFFFFFFu), 0u);
  EXPECT_EQ(mul32hi(0xFFFFFFFFu, 0xFFFFFFFFu), uint32_t((uint64_t(0xFFFFFFFFu) * 0xFFFFFFFFu) >> 32));
  EXPECT_EQ(mul32hi(0x10000u, 0x10000u), 1u); // 2^16 * 2^16 == 2^32, high half == 1
}

TEST(NcclUtilityMicrotest, Mul64Hi_ReturnsTheHighHalfOfTheFull128BitProduct) {
  EXPECT_EQ(mul64hi(0ull, ~0ull), 0ull);
  uint64_t big = 0x1'0000'0000ull; // 2^32
  EXPECT_EQ(mul64hi(big, big), 1ull); // 2^32 * 2^32 == 2^64, high half == 1
}

TEST(NcclUtilityMicrotest, ImulRcp32_HandlesZeroReciprocalShortCircuitsAndMatchesIdivRcpOfProduct) {
  // xrcp == 0 or yrcp == 0 short-circuits to the other operand's reciprocal
  // directly -- callers use this to mean "treat this factor as already 1".
  EXPECT_EQ(imulRcp32(5u, 0u, 7u, idivRcp32(7u)), idivRcp32(7u));
  EXPECT_EQ(imulRcp32(5u, idivRcp32(5u), 7u, 0u), idivRcp32(5u));

  // x=3, y=5 is chosen (rather than some arbitrary pair) because it's one of
  // the smallest inputs that actually drives the `if (x*y <= rem) rcp += 1;`
  // rounding correction true -- an arbitrary pair like (6,7) can land on the
  // false side and leave that correction untested.
  uint32_t x = 3, y = 5;
  uint32_t got = imulRcp32(x, idivRcp32(x), y, idivRcp32(y));
  EXPECT_EQ(got, idivRcp32(x * y));

  // x=2, y=2 exercises the other side of that same branch (no correction
  // needed), so both directions of the rounding check are verified.
  EXPECT_EQ(imulRcp32(2u, idivRcp32(2u), 2u, idivRcp32(2u)), idivRcp32(4u));
}

TEST(NcclUtilityMicrotest, ImulRcp64_HandlesZeroReciprocalShortCircuitsAndMatchesIdivRcpOfProduct) {
  EXPECT_EQ(imulRcp64(5ull, 0ull, 7ull, idivRcp64(7ull)), idivRcp64(7ull));
  EXPECT_EQ(imulRcp64(5ull, idivRcp64(5ull), 7ull, 0ull), idivRcp64(5ull));

  // Same rationale as the 32-bit test above: (3,5) drives the rounding
  // correction branch true, which a value like (6,7) does not for the 64-bit
  // reciprocal (the correction's trigger condition depends on word width).
  uint64_t x = 3, y = 5;
  uint64_t got = imulRcp64(x, idivRcp64(x), y, idivRcp64(y));
  EXPECT_EQ(got, idivRcp64(x * y));

  // (2,2) exercises the other side of that same branch (no correction needed).
  EXPECT_EQ(imulRcp64(2ull, idivRcp64(2ull), 2ull, idivRcp64(2ull)), idivRcp64(4ull));
}

TEST(NcclUtilityMicrotest, IdivmodFast32_MatchesOrdinaryDivisionAcrossDivisorsAndYrcpZeroShortcut) {
  for (uint32_t y : {1u, 2u, 3u, 7u, 16u, 100u, 4096u}) {
    uint32_t yrcp = idivRcp32(y);
    for (uint32_t x : {0u, 1u, 5u, 41u, 4095u, 4096u, 1000000u}) {
      uint32_t q, r;
      idivmodFast32(&q, &r, x, y, yrcp);
      EXPECT_EQ(q, x / y) << "x=" << x << " y=" << y;
      EXPECT_EQ(r, x % y) << "x=" << x << " y=" << y;
    }
  }
  // yrcp == 0 shortcut: q = x directly (caller's contract for "divisor is 1").
  uint32_t q, r;
  idivmodFast32(&q, &r, 42u, 1u, 0u);
  EXPECT_EQ(q, 42u);
  EXPECT_EQ(r, 0u);
}

TEST(NcclUtilityMicrotest, IdivmodFast64_MatchesOrdinaryDivisionAcrossDivisorsAndYrcpZeroShortcut) {
  for (uint64_t y : {1ull, 2ull, 3ull, 7ull, 16ull, 100ull, 1ull << 20}) {
    uint64_t yrcp = idivRcp64(y);
    for (uint64_t x : {0ull, 1ull, 5ull, 41ull, (1ull << 20) - 1, 1ull << 20, 1ull << 40}) {
      uint64_t q, r;
      idivmodFast64(&q, &r, x, y, yrcp);
      EXPECT_EQ(q, x / y) << "x=" << x << " y=" << y;
      EXPECT_EQ(r, x % y) << "x=" << x << " y=" << y;
    }
  }
  uint64_t q, r;
  idivmodFast64(&q, &r, 42ull, 1ull, 0ull);
  EXPECT_EQ(q, 42ull);
  EXPECT_EQ(r, 0ull);
}

TEST(NcclUtilityMicrotest, IdivFast32_MatchesRefDivisionAndIdivFast64TruncatesToLow32Bits) {
  for (uint32_t y : {1u, 3u, 16u, 4096u}) {
    uint32_t yrcp = idivRcp32(y);
    for (uint32_t x : {0u, 41u, 4095u, 1000000u}) {
      EXPECT_EQ(idivFast32(x, y, yrcp), refDiv32(x, y));
    }
  }
  // idivFast64 returns a uint32_t truncation of a 64-bit quotient; pick a
  // quotient that fits in 32 bits so truncation is a correctness check, not
  // a free pass.
  uint64_t y = 1000, yrcp64 = idivRcp64(y);
  uint64_t x = 123456789ull;
  EXPECT_EQ(idivFast64(x, y, yrcp64), uint32_t(refDiv64(x, y)));
}

TEST(NcclUtilityMicrotest, ImodFast32And64_MatchRefModulo) {
  for (uint32_t y : {1u, 3u, 16u, 4096u}) {
    uint32_t yrcp = idivRcp32(y);
    for (uint32_t x : {0u, 41u, 4095u, 1000000u}) {
      EXPECT_EQ(imodFast32(x, y, yrcp), x % y);
    }
  }
  for (uint64_t y : {1ull, 3ull, 16ull, 1ull << 20}) {
    uint64_t yrcp = idivRcp64(y);
    for (uint64_t x : {0ull, 41ull, (1ull << 20) - 1, 1ull << 40}) {
      EXPECT_EQ(imodFast64(x, y, yrcp), x % y);
    }
  }
}

TEST(NcclUtilityMicrotest, Optional_DefaultAndAbsentConstructLeavePresentFalse) {
  Optional<int> a;
  EXPECT_FALSE(a.present);
  Optional<int> b{nccl::utility::Absent{}};
  EXPECT_FALSE(b.present);
}

TEST(NcclUtilityMicrotest, Optional_PresentConstructorForwardsArgsAndSetsPresentTrue) {
  // present(1, 2) packs two ints into a Present<int&&, int&&>; Optional's
  // Present-taking constructor must forward both through to T's constructor
  // in the right order, not just construct a default T.
  struct Pair {
    int a, b;
    NCCL_HOST_DEVICE_INLINE Pair(int a_, int b_) : a(a_), b(b_) {}
  };
  Optional<Pair> opt(present(3, 4));
  ASSERT_TRUE(opt.present);
  EXPECT_EQ(opt.thing.a, 3);
  EXPECT_EQ(opt.thing.b, 4);
}

TEST(NcclUtilityMicrotest, Optional_DestructorOnlyRunsWhenPresentIsTrue) {
  static int destructorCalls = 0;
  struct Counted {
    NCCL_HOST_DEVICE_INLINE Counted() {}
    NCCL_HOST_DEVICE_INLINE ~Counted() { destructorCalls++; }
  };

  destructorCalls = 0;
  {
    Optional<Counted> absent; // present == false: ~Counted() must NOT run.
  }
  EXPECT_EQ(destructorCalls, 0);

  destructorCalls = 0;
  {
    Optional<Counted> present_opt(present());
  }
  EXPECT_EQ(destructorCalls, 1);
}

TEST(NcclUtilityMicrotest, Add4G_AddsDeltaIntoTheHighHalfOfA64BitValueViaUnion) {
  // add4G<T> reinterprets a T (must be 8 bytes) as two uint32_t halves and
  // adds delta4G into the high half (index 1 is the high word on this
  // little-endian build). Verify against plain 64-bit arithmetic.
  uint64_t base = 0x0000'0000'1234'5678ull;
  EXPECT_EQ(add4G(base, 1), base + (uint64_t(1) << 32));
  EXPECT_EQ(add4G(base, 5), base + (uint64_t(5) << 32));
  // Negative delta: subtracts a 4G chunk from the high half.
  uint64_t highBase = 0x0000'0003'0000'0000ull;
  EXPECT_EQ(add4G(highBase, -1), highBase - (uint64_t(1) << 32));
}

TEST(NcclUtilityMicrotest, AcquireOrderOfStd_DowngradesReleaseAndAcqRelPassesEverythingElseThrough) {
  EXPECT_EQ(acquireOrderOf(std::memory_order_release), std::memory_order_relaxed);
  EXPECT_EQ(acquireOrderOf(std::memory_order_acq_rel), std::memory_order_acquire);
  EXPECT_EQ(acquireOrderOf(std::memory_order_relaxed), std::memory_order_relaxed);
  EXPECT_EQ(acquireOrderOf(std::memory_order_acquire), std::memory_order_acquire);
  EXPECT_EQ(acquireOrderOf(std::memory_order_seq_cst), std::memory_order_seq_cst);
}

TEST(NcclUtilityMicrotest, ReleaseOrderOfStd_DowngradesAcquireAndAcqRelPassesEverythingElseThrough) {
  EXPECT_EQ(releaseOrderOf(std::memory_order_acquire), std::memory_order_relaxed);
  EXPECT_EQ(releaseOrderOf(std::memory_order_acq_rel), std::memory_order_release);
  EXPECT_EQ(releaseOrderOf(std::memory_order_relaxed), std::memory_order_relaxed);
  EXPECT_EQ(releaseOrderOf(std::memory_order_release), std::memory_order_release);
  EXPECT_EQ(releaseOrderOf(std::memory_order_seq_cst), std::memory_order_seq_cst);
}

TEST(NcclUtilityMicrotest, ToAtomicBuiltinOrder_MapsEveryStdOrderAndFallsBackToSeqCstForConsume) {
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_relaxed), __ATOMIC_RELAXED);
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_acquire), __ATOMIC_ACQUIRE);
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_release), __ATOMIC_RELEASE);
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_acq_rel), __ATOMIC_ACQ_REL);
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_seq_cst), __ATOMIC_SEQ_CST);
  // memory_order_consume has no case label; must fall through to default.
  EXPECT_EQ(toAtomicBuiltinOrder(std::memory_order_consume), __ATOMIC_SEQ_CST);
}

TEST(NcclUtilityMicrotest, AcquireOrderOfCuda_DowngradesReleaseAndAcqRelPassesEverythingElseThrough) {
  EXPECT_EQ(acquireOrderOf(cuda::memory_order_release), cuda::memory_order_relaxed);
  EXPECT_EQ(acquireOrderOf(cuda::memory_order_acq_rel), cuda::memory_order_acquire);
  EXPECT_EQ(acquireOrderOf(cuda::memory_order_relaxed), cuda::memory_order_relaxed);
  EXPECT_EQ(acquireOrderOf(cuda::memory_order_acquire), cuda::memory_order_acquire);
  EXPECT_EQ(acquireOrderOf(cuda::memory_order_seq_cst), cuda::memory_order_seq_cst);
}

TEST(NcclUtilityMicrotest, ReleaseOrderOfCuda_DowngradesAcquireAndAcqRelPassesEverythingElseThrough) {
  EXPECT_EQ(releaseOrderOf(cuda::memory_order_acquire), cuda::memory_order_relaxed);
  EXPECT_EQ(releaseOrderOf(cuda::memory_order_acq_rel), cuda::memory_order_release);
  EXPECT_EQ(releaseOrderOf(cuda::memory_order_relaxed), cuda::memory_order_relaxed);
  EXPECT_EQ(releaseOrderOf(cuda::memory_order_release), cuda::memory_order_release);
  EXPECT_EQ(releaseOrderOf(cuda::memory_order_seq_cst), cuda::memory_order_seq_cst);
}

TEST(NcclUtilityMicrotest, ToCudaOrder_MapsEveryStdOrderAndFallsBackToSeqCstForConsume) {
  EXPECT_EQ(toCudaOrder(std::memory_order_relaxed), cuda::memory_order_relaxed);
  EXPECT_EQ(toCudaOrder(std::memory_order_acquire), cuda::memory_order_acquire);
  EXPECT_EQ(toCudaOrder(std::memory_order_release), cuda::memory_order_release);
  EXPECT_EQ(toCudaOrder(std::memory_order_acq_rel), cuda::memory_order_acq_rel);
  EXPECT_EQ(toCudaOrder(std::memory_order_seq_cst), cuda::memory_order_seq_cst);
  EXPECT_EQ(toCudaOrder(std::memory_order_consume), cuda::memory_order_seq_cst);
}

TEST(NcclUtilityMicrotest, Present_GetByIndexReturnsEachArgumentInDeclarationOrder) {
  // Pass lvalues so H deduces to int& (present(H&&, ...) with an lvalue arg
  // collapses to a reference to the original), keeping p's stored references
  // valid for the lifetime of a,b,c below -- a prvalue-literal argument would
  // deduce H=int and leave p holding a dangling int&& into a temporary that
  // expires at the end of the present(...) call's full expression.
  int a = 10, b = 20, c = 30;
  auto p = present(a, b, c);
  EXPECT_EQ(p.get(nccl::utility::IntSeq<0>{}), 10);
  EXPECT_EQ(p.get(nccl::utility::IntSeq<1>{}), 20);
  EXPECT_EQ(p.get(nccl::utility::IntSeq<2>{}), 30);
}
