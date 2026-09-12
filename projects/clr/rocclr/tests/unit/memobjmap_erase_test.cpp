/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for amd::EraseCoveringMemObj / amd::EraseCoveringMemObjIf -- the
// range-aware erase primitive shared by MemObjMap::FindAndRemoveMemObj and the
// identity-checked MemObjMap::TryRemoveMemObj.
//
// These pin two invariants:
//
// 1. Removal uses the same [base, base + size) range test as lookup
//    (MemObjMap::FindMemObj), so any pointer a lookup resolves -- including an
//    interior address, not just the exact base key -- is removable. A true
//    miss (no covering allocation) returns nullptr and erases nothing.
//
// 2. The predicate variant erases only when the covering entry satisfies the
//    caller's check. TryRemoveMemObj passes an identity predicate so that a
//    pointer whose range happens to be covered by an *unrelated* allocation
//    (per-device VA ranges can numerically overlap the global map's entries on
//    Windows) never de-indexes that live allocation.

#include "device/memobjmap_erase.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>

namespace {

// Stand-in for amd::Memory: an allocation spanning [base, base + size).
struct FakeMem {
  uintptr_t base;
  size_t size;
};

using Map = std::map<uintptr_t, FakeMem*>;

// Size accessor mirroring MemObjMap's: maps a stored value to its byte span.
auto sizeOf = [](FakeMem* m) { return m->size; };

// Removing by the exact base key erases that entry and returns it.
TEST(EraseCoveringMemObjTest, ExactBaseKeyRemovesEntry) {
  FakeMem a{0x1000, 0x1000};
  FakeMem b{0x4000, 0x1000};
  Map map{{a.base, &a}, {b.base, &b}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x1000), sizeOf);

  EXPECT_EQ(removed, &a);
  EXPECT_EQ(map.count(0x1000), 0u);
  EXPECT_EQ(map.size(), 1u);
}

// The asymmetry fix: an interior pointer (inside [base, base + size)) that
// lookup would resolve is removable too, not only the exact base key.
TEST(EraseCoveringMemObjTest, InteriorPointerRemovesCoveringEntry) {
  FakeMem a{0x1000, 0x1000};
  Map map{{a.base, &a}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x1500), sizeOf);

  EXPECT_EQ(removed, &a);
  EXPECT_TRUE(map.empty());
}

// A pointer at base + size is one past the allocation: no entry covers it.
TEST(EraseCoveringMemObjTest, PointerAtEndOfRangeReturnsNull) {
  FakeMem a{0x1000, 0x1000};
  Map map{{a.base, &a}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x2000), sizeOf);

  EXPECT_EQ(removed, nullptr);
  EXPECT_EQ(map.size(), 1u);  // untouched
}

// A pointer in the gap between two allocations matches neither.
TEST(EraseCoveringMemObjTest, PointerInGapReturnsNull) {
  FakeMem a{0x1000, 0x1000};  // [0x1000, 0x2000)
  FakeMem b{0x4000, 0x1000};  // [0x4000, 0x5000)
  Map map{{a.base, &a}, {b.base, &b}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x3000), sizeOf);

  EXPECT_EQ(removed, nullptr);
  EXPECT_EQ(map.size(), 2u);
}

// A pointer below the lowest base has no predecessor entry.
TEST(EraseCoveringMemObjTest, PointerBelowAllReturnsNull) {
  FakeMem a{0x1000, 0x1000};
  Map map{{a.base, &a}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x500), sizeOf);

  EXPECT_EQ(removed, nullptr);
  EXPECT_EQ(map.size(), 1u);
}

// The degenerate miss: an empty map yields nullptr and never aborts.
TEST(EraseCoveringMemObjTest, EmptyMapReturnsNull) {
  Map map;

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x1), sizeOf);

  EXPECT_EQ(removed, nullptr);
  EXPECT_TRUE(map.empty());
}

// With several allocations, the correct covering entry is the one removed.
TEST(EraseCoveringMemObjTest, SelectsCorrectCoveringEntryAmongMany) {
  FakeMem a{0x1000, 0x1000};
  FakeMem b{0x2000, 0x1000};
  FakeMem c{0x3000, 0x1000};
  Map map{{a.base, &a}, {b.base, &b}, {c.base, &c}};

  FakeMem* removed = amd::EraseCoveringMemObj(map, uintptr_t(0x2abc), sizeOf);

  EXPECT_EQ(removed, &b);
  EXPECT_EQ(map.count(0x2000), 0u);
  EXPECT_EQ(map.size(), 2u);
}

// When the covering entry is the object the caller expects, it is erased --
// TryRemoveMemObj's happy path.
TEST(EraseCoveringMemObjIfTest, MatchingPredicateErases) {
  FakeMem a{0x1000, 0x1000};
  Map map{{a.base, &a}};
  const FakeMem* expected = &a;

  FakeMem* removed = amd::EraseCoveringMemObjIf(
      map, uintptr_t(0x1500), sizeOf, [expected](const FakeMem* m) { return m == expected; });

  EXPECT_EQ(removed, &a);
  EXPECT_TRUE(map.empty());
}

// The wrong-erase regression test: a pointer covered by an *unrelated*
// allocation (overlapping per-device VA on Windows) must not de-index that
// live entry when the caller expected a different object.
TEST(EraseCoveringMemObjIfTest, RejectingPredicateLeavesEntry) {
  FakeMem a{0x1000, 0x1000};
  FakeMem other{0x9000, 0x1000};  // what the caller actually resolved
  Map map{{a.base, &a}};
  const FakeMem* expected = &other;

  FakeMem* removed = amd::EraseCoveringMemObjIf(
      map, uintptr_t(0x1500), sizeOf, [expected](const FakeMem* m) { return m == expected; });

  EXPECT_EQ(removed, nullptr);
  EXPECT_EQ(map.count(0x1000), 1u);  // a survives untouched
}

// The predicate is consulted only for a covering entry; a range miss returns
// nullptr without ever invoking it.
TEST(EraseCoveringMemObjIfTest, PredicateNotInvokedWithoutCoveringEntry) {
  FakeMem a{0x1000, 0x1000};
  Map map{{a.base, &a}};
  int calls = 0;

  FakeMem* removed = amd::EraseCoveringMemObjIf(map, uintptr_t(0x3000), sizeOf,
                                                [&calls](const FakeMem*) {
                                                  ++calls;
                                                  return true;
                                                });

  EXPECT_EQ(removed, nullptr);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(map.size(), 1u);
}

}  // namespace
