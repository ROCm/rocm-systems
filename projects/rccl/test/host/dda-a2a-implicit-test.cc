/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the bounds of the DDA AllToAll implicit registration cache
// (src/algorithms/dda/alltoall/dda_alltoall_implicit_policy.h): graph-destructor unpins, the cache
// cap with pinned entries, and the legacy (hipMalloc) import byte budget.

#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <vector>

#include "algorithms/dda/alltoall/dda_alltoall_implicit_policy.h"

namespace {

using namespace dda_a2a_implicit;

struct Entry {
  uint64_t uid;
  int pins;
  uint64_t lastUse;
};

}  // namespace

// A destroyed graph releases exactly the pins its capture took, wherever the entry now lives
// (cache, or zombies after a free + reallocation at the same address).
TEST(DdaA2aImplicitMicrotest, GraphDestructorUnpinsItsEntry) {
  auto q = std::make_shared<UnpinQueue>();
  Entry a{1, 2, 0}, b{2, 1, 0}, z{3, 1, 0};
  std::vector<Entry*> cache{&a, &b}, zombies{&z};
  // two captures of `a`, one of `z`; destructors run on another thread (as the runtime's would)
  std::thread t([&] {
    unpinDestructor(new UnpinArg{q, 1});
    unpinDestructor(new UnpinArg{q, 3});
  });
  t.join();
  applyUnpins(*q, cache, zombies);
  EXPECT_EQ(1, a.pins);
  EXPECT_EQ(1, b.pins);
  EXPECT_EQ(0, z.pins);
  unpinDestructor(new UnpinArg{q, 1});
  unpinDestructor(new UnpinArg{q, 99});  // entry already gone: ignored
  applyUnpins(*q, cache, zombies);
  EXPECT_EQ(0, a.pins);
  applyUnpins(*q, cache, zombies);  // nothing queued: no change, never below zero
  EXPECT_EQ(0, a.pins);
  EXPECT_TRUE(q->uids.empty());
}

// A graph destroyed after its communicator: the destructor only touches the shared queue, which
// the destructor object keeps alive.
TEST(DdaA2aImplicitMicrotest, GraphOutlivingTheCommIsSafe) {
  auto* arg = new UnpinArg{std::make_shared<UnpinQueue>(), 7};
  std::weak_ptr<UnpinQueue> w = arg->q;
  // the comm (sole other owner) is gone here; only the destructor object holds the queue
  EXPECT_FALSE(w.expired());
  unpinDestructor(arg);
  EXPECT_TRUE(w.expired());
}

// At the cap, only an unpinned entry may be evicted (least recently used); with every entry pinned
// the new allocation must be staged instead of growing the cache.
TEST(DdaA2aImplicitMicrotest, CacheCapNeverExceededByPinnedEntries) {
  Entry e0{1, 0, 50}, e1{2, 1, 10}, e2{3, 0, 30}, e3{4, 2, 5};
  std::vector<Entry*> cache{&e0, &e1, &e2, &e3};
  EXPECT_EQ(2u, lruUnpinned(cache));  // e2: oldest unpinned (e1, e3 older but pinned)
  e2.pins = 1;
  EXPECT_EQ(0u, lruUnpinned(cache));
  e0.pins = 1;
  EXPECT_EQ(cache.size(), lruUnpinned(cache));  // all pinned: refuse
  std::vector<Entry*> empty;
  EXPECT_EQ(0u, lruUnpinned(empty));
}

TEST(DdaA2aImplicitMicrotest, LegacyBudget) {
  constexpr int64_t kGiB = 1ll << 30;
  // default 4 GiB per comm
  EXPECT_TRUE(legacyAllowed(4 * kGiB, 0, 4 * kGiB));
  EXPECT_TRUE(legacyAllowed(4 * kGiB, 3 * kGiB, kGiB));
  EXPECT_FALSE(legacyAllowed(4 * kGiB, 3 * kGiB, kGiB + 1));
  EXPECT_FALSE(legacyAllowed(4 * kGiB, 0, 4 * kGiB + 1));
  // 0: never import hipMalloc memory
  EXPECT_FALSE(legacyAllowed(0, 0, 1));
  // < 0: unlimited (the pre-budget behaviour)
  EXPECT_TRUE(legacyAllowed(-1, size_t(64) * kGiB, size_t(64) * kGiB));
  // no wrap-around on huge sizes
  EXPECT_FALSE(legacyAllowed(4 * kGiB, 1, SIZE_MAX));
}

// Free + reallocate churn of hipMalloc buffers: imported bytes plateau at the budget (the freed
// allocations stay imported -- they cannot be closed -- so the budget bounds the pinned memory).
TEST(DdaA2aImplicitMicrotest, LegacyChurnPlateausAtBudget) {
  constexpr int64_t kBudget = 256ll << 20;
  constexpr size_t kBuf = 64u << 20;
  size_t imported = 0;
  int staged = 0;
  for (int it = 0; it < 1000; ++it) {
    if (legacyAllowed(kBudget, imported, kBuf)) {
      imported += kBuf;
    } else {
      ++staged;
    }
  }
  EXPECT_EQ(size_t(kBudget), imported);
  EXPECT_EQ(1000 - 4, staged);
}
