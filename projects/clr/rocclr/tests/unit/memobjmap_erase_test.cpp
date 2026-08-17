/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for the mem-obj map erase primitives in device/memobjmap_erase.hpp.
//
// These pin two invariants:
//
// 1. EraseCoveringMemObj (behind MemObjMap::FindAndRemoveMemObj) removes with
//    the same [base, base + size) range test lookup uses (MemObjMap::FindMemObj),
//    so any pointer a lookup resolves -- including an interior address, not just
//    the exact base key -- is removable. A true miss (no covering allocation)
//    returns nullptr and erases nothing.
//
// 2. The identity-sweep primitives (behind MemObjMap::TryRemoveMemObj) let a
//    user-facing free drop *every* alias an allocation is indexed under before
//    the object is released (EraseEntriesWithValue), while distinguishing a
//    base address the allocation is actually indexed under from an interior
//    pointer or a numerically-overlapping unrelated allocation
//    (ContainsKeyWithValue / ContainsValue), which must never be de-indexed
//    or freed by mistake.

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

// A free must drop every alias of the object it releases: an allocation
// indexed under two keys (e.g. a registered host pointer plus a device VA)
// loses both, and unrelated entries survive.
TEST(EraseEntriesWithValueTest, ErasesEveryAliasOfValue) {
  FakeMem a{0x1000, 0x1000};
  FakeMem b{0x4000, 0x1000};
  Map map{{0x1000, &a}, {0x8000, &a}, {b.base, &b}};

  size_t erased = amd::EraseEntriesWithValue(map, &a);

  EXPECT_EQ(erased, 2u);
  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.count(0x4000), 1u);  // b untouched
}

// Sweeping for an object the map does not hold erases nothing.
TEST(EraseEntriesWithValueTest, AbsentValueErasesNothing) {
  FakeMem a{0x1000, 0x1000};
  FakeMem other{0x9000, 0x1000};
  Map map{{a.base, &a}};

  EXPECT_EQ(amd::EraseEntriesWithValue(map, &other), 0u);
  EXPECT_EQ(map.size(), 1u);
}

// Tracking is detected under any key the object is indexed by, not just its
// own base -- and never for an object the map does not hold.
TEST(ContainsValueTest, FindsValueUnderAnyKey) {
  FakeMem a{0x1000, 0x1000};
  FakeMem other{0x9000, 0x1000};
  Map map{{0x8000, &a}};  // indexed under an alias, not its own base

  EXPECT_TRUE(amd::ContainsValue(map, &a));
  EXPECT_FALSE(amd::ContainsValue(map, &other));
}

// The base-address test behind rejecting bad frees: an exact key owned by the
// object passes; an interior pointer (covered by the object's range but not a
// key) and another object's key both fail.
TEST(ContainsKeyWithValueTest, RequiresExactKeyAndIdentity) {
  FakeMem a{0x1000, 0x1000};
  FakeMem b{0x2000, 0x1000};
  Map map{{a.base, &a}, {b.base, &b}};

  EXPECT_TRUE(amd::ContainsKeyWithValue(map, uintptr_t(0x1000), &a));
  EXPECT_FALSE(amd::ContainsKeyWithValue(map, uintptr_t(0x1500), &a));
  EXPECT_FALSE(amd::ContainsKeyWithValue(map, uintptr_t(0x2000), &a));
}

}  // namespace
