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

// Staging slot of pairs that are not mapped (64 MiB staging buffer per rank, the production size):
// one 4 KiB-multiple slot per source, never above the per-peer chunk rounded up to 4 KiB, and a
// round count every rank computes alike. The Direct window (P > 1 MiB, T <= 16 GiB) stays far below
// the 16-bit round limit.
TEST(DdaA2aImplicitMicrotest, StagingSlotAndRounds) {
  const size_t M = 1u << 20;
  const size_t scratch = 64 * M;
  EXPECT_EQ(8 * M, stagingSlot(scratch, 8, 1ull << 30));
  EXPECT_EQ(4 * M, stagingSlot(scratch, 16, 1ull << 30));
  EXPECT_EQ(32 * M, stagingSlot(scratch, 2, 8ull << 30));
  EXPECT_EQ(2 * M, stagingSlot(scratch, 8, 2 * M));                    // capped at the chunk
  EXPECT_EQ(2 * M + 4096, stagingSlot(scratch, 8, 2 * M + 16));        // chunk rounded up to 4 KiB
  EXPECT_EQ(size_t(5461) * 4096, stagingSlot(scratch, 3, 1ull << 30));  // non-power-of-two rank count
  EXPECT_EQ(0u, stagingSlot(scratch, 0, M));
  EXPECT_EQ(0u, stagingSlot(4096 * 4 - 1, 4, M));                      // under 4 KiB per source
  EXPECT_EQ(SIZE_MAX, stagingRounds(4096 * 4 - 1, 4, M));
  EXPECT_EQ(1u, stagingRounds(scratch, 8, 2 * M));
  EXPECT_EQ(128u, stagingRounds(scratch, 8, 1ull << 30));             // 8r, T = 8 GiB
  EXPECT_EQ(256u, stagingRounds(scratch, 16, 1ull << 30));            // 16r, T = 16 GiB (a2aDirectMax)
  EXPECT_EQ(256u, stagingRounds(scratch, 2, 8ull << 30));             // 2r, T = 16 GiB
  EXPECT_EQ(1u, stagingRounds(scratch, 16, 1));
}

namespace {
struct Retiring {
  void* ev;
  int id;
};
struct LegacyEntry {
  uintptr_t base;
  bool legacy;
};
}  // namespace

// Retiring entries never query their events while the call's stream is capturing (a query during a
// capture invalidates it); only a definite completion lets an entry close, any other query result
// (not ready, an error such as a capture-unsupported return) keeps it.
TEST(DdaA2aImplicitMicrotest, ProgressNeverQueriesWhileCapturing) {
  int ev0 = 0, ev1 = 0;
  std::vector<Retiring> v = {{&ev0, 0}, {&ev1, 1}, {nullptr, 2}};
  int queries = 0, closes = 0;
  auto query = [&](Retiring& r) {
    ++queries;
    if (r.id == 0) return false;  // not ready, or any error
    r.ev = nullptr;
    return true;
  };
  auto close = [&](Retiring&) {
    ++closes;
    return true;
  };
  progressRetiring(v, /*capturing=*/true, query, close);
  EXPECT_EQ(0, queries);
  EXPECT_EQ(0, closes);
  EXPECT_EQ(3u, v.size());
  progressRetiring(v, /*capturing=*/false, query, close);
  EXPECT_EQ(2, queries);  // entry 2 has no event left to wait for
  EXPECT_EQ(2, closes);
  ASSERT_EQ(1u, v.size());
  EXPECT_EQ(0, v[0].id);
  // A close still in flight keeps the entry (its event is not queried again).
  v[0].ev = nullptr;
  progressRetiring(v, false, query, [](Retiring&) { return false; });
  EXPECT_EQ(1u, v.size());
  EXPECT_EQ(2, queries);
}

// An ncclCommRegister record of an allocation the cache already imported with legacy IPC (cached or
// retired-but-mapped) must keep using the cache (no second legacy import).
TEST(DdaA2aImplicitMicrotest, LegacyEntryAtBasePreferredOverRegistration) {
  LegacyEntry a{0x1000, true}, b{0x2000, false}, z{0x3000, true};
  std::vector<LegacyEntry*> cache = {&a, &b}, zombies = {&z};
  EXPECT_TRUE(hasLegacyAt(cache, zombies, 0x1000));
  EXPECT_FALSE(hasLegacyAt(cache, zombies, 0x2000));  // VMM import: the registration may import it again
  EXPECT_TRUE(hasLegacyAt(cache, zombies, 0x3000));
  EXPECT_FALSE(hasLegacyAt(cache, zombies, 0x4000));
}

// The real implicit predicate (ncclAllToAllDdaFabricDirectImplicitEligible forwards to it): the agreed
// staging part, shape and round bound decide; registration state is not an input.
TEST(DdaA2aImplicitMicrotest, ImplicitEligiblePredicate) {
  const size_t M = 1u << 20, S = 64 * M, maxR = 0xFFFE;
  const uintptr_t snd = 0x10000000, rcv = 0x20000000;
  EXPECT_TRUE(implicitEligible(true, S, 8, 8, 16, 2 * M, snd, rcv, true, maxR));
  EXPECT_FALSE(implicitEligible(false, S, 8, 8, 16, 2 * M, snd, rcv, true, maxR));  // no staging buffer agreed
  EXPECT_FALSE(implicitEligible(true, S, 8, 8, 16, 2 * M, rcv, rcv, true, maxR));   // in place
  EXPECT_FALSE(implicitEligible(true, S, 32, 32, 16, 2 * M, snd, rcv, true, maxR)); // > 16 ranks
  EXPECT_FALSE(implicitEligible(true, S, 8, 4, 16, 2 * M, snd, rcv, true, maxR));   // not one LSA team
  EXPECT_FALSE(implicitEligible(true, S, 8, 8, 16, 2 * M + 8, snd, rcv, true, maxR));  // P % 16
  EXPECT_FALSE(implicitEligible(true, S, 8, 8, 16, 2 * M, snd + 8, rcv, true, maxR));  // alignment
  EXPECT_FALSE(implicitEligible(true, S, 8, 8, 16, 2 * M, snd, rcv, false, maxR));     // no geometry
  EXPECT_FALSE(implicitEligible(true, 4096 * 8 - 1, 8, 8, 16, 2 * M, snd, rcv, true, maxR));  // no slot
  EXPECT_TRUE(implicitEligible(true, S, 8, 8, 16, 1ull << 30, snd, rcv, true, 128));   // 128 rounds
  EXPECT_FALSE(implicitEligible(true, S, 8, 8, 16, 1ull << 30, snd, rcv, true, 127));  // round bound
}

// The launcher's kernel choice: window kernel for windowed calls, hybrid otherwise.
TEST(DdaA2aImplicitMicrotest, DispatchDirectByWindow) {
  int w = 0, h = 0;
  EXPECT_EQ(1, dispatchDirect(true, [&] { return ++w; }, [&] { return ++h + 100; }));
  EXPECT_EQ(101, dispatchDirect(false, [&] { return ++w; }, [&] { return ++h + 100; }));
  EXPECT_EQ(1, w);
  EXPECT_EQ(1, h);
}
