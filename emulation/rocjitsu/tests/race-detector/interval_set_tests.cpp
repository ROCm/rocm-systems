// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include <gtest/gtest.h>

using namespace rocjitsu::plugins::race_detector;

TEST(IntervalSet, EmptyByDefault) {
  IntervalSet s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);
  EXPECT_EQ(s.getTotalBytes(), 0);
}

TEST(IntervalSet, SingleAppend) {
  IntervalSet s;
  s.append(10, 14);
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 4);
  EXPECT_TRUE(s.contains(10));
  EXPECT_TRUE(s.contains(13));
  EXPECT_FALSE(s.contains(9));
  EXPECT_FALSE(s.contains(14));
}

TEST(IntervalSet, AdjacentAppendMerges) {
  IntervalSet s;
  s.append(0, 4);
  s.append(4, 8);
  // Adjacent intervals should merge on append.
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 8);
  auto it = s.begin();
  EXPECT_EQ(it->start, 0);
  EXPECT_EQ(it->end, 8);
}

TEST(IntervalSet, OverlappingAppendMerges) {
  IntervalSet s;
  s.append(0, 6);
  s.append(4, 10);
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 10);
}

TEST(IntervalSet, NonAdjacentAppendDoesNotMerge) {
  IntervalSet s;
  s.append(0, 4);
  s.append(8, 12);
  EXPECT_EQ(s.size(), 2u);
  EXPECT_EQ(s.getTotalBytes(), 8);
}

TEST(IntervalSet, OverlappingAppendMergesWithBack) {
  // [5,10) then [0,7) overlap → merge to [0,10).
  // [20,25) is disjoint → new entry.
  // [15,22) overlaps [20,25) → merge to [15,25).
  IntervalSet s;
  s.append(5, 10);
  s.append(0, 7);
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.begin()->start, 0);
  EXPECT_EQ(s.begin()->end, 10);

  s.append(20, 25);
  EXPECT_EQ(s.size(), 2u);

  s.append(15, 22);
  EXPECT_EQ(s.size(), 2u);
  auto it = s.begin();
  EXPECT_EQ(it->start, 0);
  EXPECT_EQ(it->end, 10);
  ++it;
  EXPECT_EQ(it->start, 15);
  EXPECT_EQ(it->end, 25);
}

TEST(IntervalSet, FinalizeIdempotent) {
  IntervalSet s;
  s.append(0, 4);
  s.append(8, 12);
  s.finalize();
  s.finalize();
  EXPECT_EQ(s.size(), 2u);
}

TEST(IntervalSet, MonotonicAppendNoFinalizeNeeded) {
  // Simulates the common ds_load_b32 pattern: lane 0 reads [0,4),
  // lane 1 reads [4,8), etc. All adjacent, all merge on append.
  IntervalSet s;
  for (int lane = 0; lane < 32; ++lane) {
    s.append(lane * 4, lane * 4 + 4);
  }
  // Everything merged during append.
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 128);
  EXPECT_EQ(s.begin()->start, 0);
  EXPECT_EQ(s.begin()->end, 128);
}

TEST(IntervalSet, StridedNonAdjacentAccess) {
  // Stride-8 access: [0,4), [8,12), [16,20), ...
  // None should merge on append or finalize.
  IntervalSet s;
  for (int lane = 0; lane < 32; ++lane) {
    s.append(lane * 8, lane * 8 + 4);
  }
  EXPECT_EQ(s.size(), 32u);
  EXPECT_EQ(s.getTotalBytes(), 128);
  s.finalize();
  EXPECT_EQ(s.size(), 32u);
}

TEST(IntervalSet, ReverseAdjacentAppendMerges) {
  // Append in reverse order. Each new interval is adjacent to the back,
  // so append merges them all into one.
  IntervalSet s;
  for (int lane = 31; lane >= 0; --lane) {
    s.append(lane * 4, lane * 4 + 4);
  }
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 128);
  EXPECT_EQ(s.begin()->start, 0);
  EXPECT_EQ(s.begin()->end, 128);
}

TEST(IntervalSet, DisjointReverseRequiresFinalize) {
  // Append non-adjacent intervals in reverse order.
  // No merging during append; finalize sorts and merges adjacent ones.
  IntervalSet s;
  s.append(20, 24);
  s.append(10, 14);
  s.append(0, 4);
  EXPECT_EQ(s.size(), 3u);
  s.finalize();
  EXPECT_EQ(s.size(), 3u);
  auto it = s.begin();
  EXPECT_EQ(it->start, 0);
  EXPECT_EQ((it + 1)->start, 10);
  EXPECT_EQ((it + 2)->start, 20);
}

TEST(IntervalSet, ContainsAfterFinalize) {
  IntervalSet s;
  s.append(20, 24);
  s.append(0, 4);
  s.finalize();
  EXPECT_TRUE(s.contains(0));
  EXPECT_TRUE(s.contains(3));
  EXPECT_FALSE(s.contains(4));
  EXPECT_TRUE(s.contains(20));
  EXPECT_FALSE(s.contains(24));
  EXPECT_FALSE(s.contains(10));
}

TEST(IntervalSet, RangeBasedFor) {
  IntervalSet s;
  s.append(0, 4);
  s.append(8, 12);
  int total = 0;
  for (const auto &iv : s) {
    total += iv.end - iv.start;
  }
  EXPECT_EQ(total, 8);
}

TEST(IntervalSet, DuplicateAppend) {
  // Appending the same interval twice should merge.
  IntervalSet s;
  s.append(0, 4);
  s.append(0, 4);
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.getTotalBytes(), 4);
}

// --- overlapsRange tests (binary search) ---

TEST(IntervalSet, OverlapsRangeEmpty) {
  IntervalSet s;
  EXPECT_FALSE(s.overlapsRange(0, 10));
}

TEST(IntervalSet, OverlapsRangeExactMatch) {
  IntervalSet s;
  s.append(10, 20);
  s.finalize();
  EXPECT_TRUE(s.overlapsRange(10, 20));
}

TEST(IntervalSet, OverlapsRangePartialOverlap) {
  IntervalSet s;
  s.append(10, 20);
  s.finalize();
  // Query overlaps the left edge.
  EXPECT_TRUE(s.overlapsRange(5, 15));
  // Query overlaps the right edge.
  EXPECT_TRUE(s.overlapsRange(15, 25));
  // Query is a subset.
  EXPECT_TRUE(s.overlapsRange(12, 18));
  // Query is a superset.
  EXPECT_TRUE(s.overlapsRange(5, 25));
}

TEST(IntervalSet, OverlapsRangeNoOverlap) {
  IntervalSet s;
  s.append(10, 20);
  s.finalize();
  // Query is entirely before.
  EXPECT_FALSE(s.overlapsRange(0, 10));
  // Query is entirely after.
  EXPECT_FALSE(s.overlapsRange(20, 30));
  // Single byte just before.
  EXPECT_FALSE(s.overlapsRange(9, 10));
  // Single byte just after.
  EXPECT_FALSE(s.overlapsRange(20, 21));
}

TEST(IntervalSet, OverlapsRangeMultipleIntervals) {
  // [0,4) [10,14) [20,24)
  IntervalSet s;
  s.append(0, 4);
  s.append(10, 14);
  s.append(20, 24);
  s.finalize();
  // Hits first interval.
  EXPECT_TRUE(s.overlapsRange(2, 6));
  // Hits middle interval.
  EXPECT_TRUE(s.overlapsRange(12, 13));
  // Hits last interval.
  EXPECT_TRUE(s.overlapsRange(18, 22));
  // Falls in gap between first and second.
  EXPECT_FALSE(s.overlapsRange(5, 9));
  // Falls in gap between second and third.
  EXPECT_FALSE(s.overlapsRange(15, 19));
  // Before all.
  EXPECT_FALSE(s.overlapsRange(-5, 0));
  // After all.
  EXPECT_FALSE(s.overlapsRange(25, 30));
}

TEST(IntervalSet, OverlapsRangeSingleByte) {
  IntervalSet s;
  s.append(100, 104);
  s.finalize();
  // Single-byte query inside.
  EXPECT_TRUE(s.overlapsRange(100, 101));
  EXPECT_TRUE(s.overlapsRange(103, 104));
  // Single-byte query outside.
  EXPECT_FALSE(s.overlapsRange(99, 100));
  EXPECT_FALSE(s.overlapsRange(104, 105));
}
