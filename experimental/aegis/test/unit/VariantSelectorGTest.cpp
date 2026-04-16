//===-- VariantSelectorGTest.cpp ---------------------------------*- C++ -*===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for `VariantSelector` — per-kernel round-robin variant picker
/// (Phase 4a of the Instrumentation Replay 5a plan).
///
//===----------------------------------------------------------------------===//

#include "aegisbit/VariantSelector.h"

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace aegisbit;

TEST(VariantSelector, SingleVariantAlwaysZero) {
  VariantSelector VS;
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(VS.pick("k", 1), 0u);
  EXPECT_EQ(VS.pick("k", 0), 0u); // degenerate 0 also returns 0
}

TEST(VariantSelector, RoundRobinSingleThread) {
  VariantSelector VS;
  std::vector<uint32_t> Seen;
  for (int i = 0; i < 10; ++i)
    Seen.push_back(VS.pick("k", 3));
  EXPECT_EQ(Seen,
            (std::vector<uint32_t>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0}));
}

TEST(VariantSelector, DifferentKernelsHaveIndependentCounters) {
  VariantSelector VS;
  EXPECT_EQ(VS.pick("a", 3), 0u);
  EXPECT_EQ(VS.pick("a", 3), 1u);
  EXPECT_EQ(VS.pick("b", 3), 0u); // "b" starts fresh
  EXPECT_EQ(VS.pick("a", 3), 2u);
  EXPECT_EQ(VS.pick("b", 3), 1u);
}

TEST(VariantSelector, ResetKernelRestartsCounter) {
  VariantSelector VS;
  for (int i = 0; i < 5; ++i)
    (void)VS.pick("k", 3);
  VS.resetKernel("k");
  EXPECT_EQ(VS.pick("k", 3), 0u);
  EXPECT_EQ(VS.pick("k", 3), 1u);
}

TEST(VariantSelector, ResetAllRestartsAllCounters) {
  VariantSelector VS;
  (void)VS.pick("a", 3);
  (void)VS.pick("a", 3);
  (void)VS.pick("b", 2);
  VS.resetAll();
  EXPECT_EQ(VS.pick("a", 3), 0u);
  EXPECT_EQ(VS.pick("b", 2), 0u);
}

TEST(VariantSelector, VariantCountChangeRemappingIsModular) {
  VariantSelector VS;
  // Simulate a plateau where variant count shrinks: the counter keeps
  // growing so `pick` continues to rotate within the new range.
  for (int i = 0; i < 5; ++i)
    (void)VS.pick("k", 4); // 0,1,2,3,0
  // Next call with count=3: raw counter=5, 5 % 3 == 2
  EXPECT_EQ(VS.pick("k", 3), 2u);
  EXPECT_EQ(VS.pick("k", 3), 0u);
  EXPECT_EQ(VS.pick("k", 3), 1u);
}

TEST(VariantSelector, MultiThreadEveryVariantVisited) {
  VariantSelector VS;
  constexpr uint32_t VariantCount = 4;
  constexpr int Threads = 8;
  constexpr int PerThread = 200;

  std::array<std::atomic<uint32_t>, VariantCount> Hits{};
  for (auto &H : Hits)
    H.store(0);

  std::vector<std::thread> Ts;
  for (int t = 0; t < Threads; ++t) {
    Ts.emplace_back([&]() {
      for (int i = 0; i < PerThread; ++i) {
        uint32_t V = VS.pick("kernel", VariantCount);
        ASSERT_LT(V, VariantCount);
        Hits[V].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &T : Ts)
    T.join();

  // Total hits must equal total picks.
  uint32_t Total = 0;
  for (const auto &H : Hits)
    Total += H.load();
  EXPECT_EQ(Total, static_cast<uint32_t>(Threads * PerThread));

  // Each variant must get hit at least once — the atomic fetch_add guarantees
  // the counter monotonically covers every residue class mod VariantCount.
  for (uint32_t i = 0; i < VariantCount; ++i)
    EXPECT_GT(Hits[i].load(), 0u)
        << "Variant " << i << " was never selected in "
        << (Threads * PerThread) << " round-robin picks";
}
