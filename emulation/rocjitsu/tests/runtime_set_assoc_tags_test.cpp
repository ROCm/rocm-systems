// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/runtime_set_assoc_tags.h"

#include <gtest/gtest.h>

using simdojo::RuntimeSetAssocTags;

namespace {

// A 2-set, 2-way, 64B-line store. line_base values are byte addresses of line
// starts; set = (base / 64) % 2.
RuntimeSetAssocTags make() {
  RuntimeSetAssocTags t;
  t.configure(/*sets=*/2, /*ways=*/2, /*line_bytes=*/64);
  return t;
}

TEST(RuntimeSetAssocTags, ConfigureAndSlotCount) {
  RuntimeSetAssocTags t = make();
  EXPECT_TRUE(t.configured());
  EXPECT_EQ(t.slot_count(), 4u);  // 2 sets * 2 ways
}

TEST(RuntimeSetAssocTags, UnconfiguredIsNoOp) {
  RuntimeSetAssocTags t;
  t.configure(0, 4, 64);
  EXPECT_FALSE(t.configured());
  EXPECT_EQ(t.slot_count(), 0u);
  EXPECT_EQ(t.lookup(0), -1);
  EXPECT_FALSE(t.present(0));
}

TEST(RuntimeSetAssocTags, LookupMissThenInstallThenHit) {
  RuntimeSetAssocTags t = make();
  EXPECT_EQ(t.lookup(0), -1);             // cold miss
  uint32_t slot = t.victim(0);            // pick a way in set 0
  t.install(slot, 0);
  EXPECT_EQ(t.lookup(0), static_cast<int>(slot));  // now a hit, same slot
  EXPECT_TRUE(t.present(0));
}

TEST(RuntimeSetAssocTags, VictimPicksInvalidWayFirstLowestIndex) {
  RuntimeSetAssocTags t = make();
  // Set 0 = bases that map to set 0: base/64 even. 0 -> set0, 128 -> set0.
  uint32_t s0 = t.victim(0);              // both ways invalid -> lowest index
  t.install(s0, 0);
  uint32_t s1 = t.victim(128);            // one invalid way left -> the other index
  EXPECT_NE(s0, s1);
  // s0 took way 0; victim must return the sibling way (way 1) in the same 2-way set.
  EXPECT_EQ(s1, s0 ^ 1u);                 // the sibling way in the same set
}

TEST(RuntimeSetAssocTags, VictimPicksLruWhenFull) {
  RuntimeSetAssocTags t = make();
  uint32_t a = t.victim(0);   t.install(a, 0);     // installs (MRU=a)
  uint32_t b = t.victim(128); t.install(b, 128);   // installs (MRU=b), a now LRU
  uint32_t v = t.victim(256); // set0 full (0 and 128) -> evict LRU == a
  EXPECT_EQ(v, a);
}

TEST(RuntimeSetAssocTags, LookupDoesNotTouchLru) {
  RuntimeSetAssocTags t = make();
  uint32_t a = t.victim(0);   t.install(a, 0);     // MRU=a
  uint32_t b = t.victim(128); t.install(b, 128);   // MRU=b, a LRU
  EXPECT_GE(t.lookup(0), 0);                        // hit on a -- must NOT promote
  uint32_t v = t.victim(256); // still evicts a (lookup did not touch)
  EXPECT_EQ(v, a);
}

TEST(RuntimeSetAssocTags, TouchPromotesToMru) {
  RuntimeSetAssocTags t = make();
  uint32_t a = t.victim(0);   t.install(a, 0);     // MRU=a
  uint32_t b = t.victim(128); t.install(b, 128);   // MRU=b, a LRU
  t.touch(a);                                       // promote a -> b now LRU
  uint32_t v = t.victim(256);
  EXPECT_EQ(v, b);
}

TEST(RuntimeSetAssocTags, ResetInvalidatesAll) {
  RuntimeSetAssocTags t = make();
  uint32_t a = t.victim(0); t.install(a, 0);
  EXPECT_TRUE(t.present(0));
  t.reset();
  EXPECT_FALSE(t.present(0));
  EXPECT_EQ(t.lookup(0), -1);
}

TEST(RuntimeSetAssocTags, ModuloNonPowerOfTwoSets) {
  RuntimeSetAssocTags t;
  t.configure(/*sets=*/3, /*ways=*/1, /*line_bytes=*/64);
  // base/64 == 0 -> set0; == 3 (base 192) -> set0; == 1 (base 64) -> set1.
  uint32_t s0 = t.victim(0);   t.install(s0, 0);
  // base 192 maps to set 0 (3 % 3 == 0); 1-way set already full -> evicts slot s0.
  uint32_t v = t.victim(192);
  EXPECT_EQ(v, s0);
  // base 64 maps to set 1 -> a different slot.
  uint32_t s1 = t.victim(64);
  EXPECT_NE(s1, s0);
}

TEST(RuntimeSetAssocTags, ReconfigureResizesAndClears) {
  RuntimeSetAssocTags t = make();        // 2x2
  uint32_t a = t.victim(0); t.install(a, 0);
  EXPECT_TRUE(t.present(0));
  t.configure(/*sets=*/4, /*ways=*/1, /*line_bytes=*/64);  // re-geometry
  EXPECT_EQ(t.slot_count(), 4u);
  EXPECT_FALSE(t.present(0));             // old residency gone after reconfigure
}

}  // namespace
