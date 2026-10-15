/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/rings.cc: turning per-rank next/prev links into ring orders.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "fakes/graph_fakes.h"

#include RINGS_CC_PATH

namespace {

class RingsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetGraphFakes(); }
  void TearDown() override { ResetGraphFakes(); }
};

// next[] and prev[] for rings visiting the ranks in each given order, channel after channel.
struct Links {
  std::vector<int> next, prev;
};

Links LinksOf(const std::vector<std::vector<int>>& orders) {
  Links l;
  for (const std::vector<int>& order : orders) {
    int n = order.size();
    std::vector<int> next(n), prev(n);
    for (int i = 0; i < n; i++) {
      next[order[i]] = order[(i + 1) % n];
      prev[order[(i + 1) % n]] = order[i];
    }
    l.next.insert(l.next.end(), next.begin(), next.end());
    l.prev.insert(l.prev.end(), prev.begin(), prev.end());
  }
  return l;
}

TEST_F(RingsMicrotest, BuildRings_WalksNextFromTheRank) {
  Links l = LinksOf({{0, 1, 2, 3}, {0, 2, 1, 3}});
  std::vector<int> rings(8, -1);

  ASSERT_EQ(ncclSuccess, ncclBuildRings(2, rings.data(), 2, 4, l.prev.data(), l.next.data()));

  EXPECT_EQ(std::vector<int>({2, 3, 0, 1, 2, 1, 3, 0}), rings);
  EXPECT_FALSE(Logged("Channel 00/02"));  // only rank 0 prints the rings
}

TEST_F(RingsMicrotest, BuildRings_RankZeroLogsEachChannel) {
  Links l = LinksOf({{0, 1, 2, 3}, {0, 2, 1, 3}});
  std::vector<int> rings(8, -1);

  ASSERT_EQ(ncclSuccess, ncclBuildRings(2, rings.data(), 0, 4, l.prev.data(), l.next.data()));

  EXPECT_TRUE(Logged("Channel 00/02 : 0 1 2 3"));
  EXPECT_TRUE(Logged("Channel 01/02 : 0 2 1 3"));
}

TEST_F(RingsMicrotest, BuildRings_ShortLoopOrOpenRingIsAnError) {
  std::vector<int> rings(4, -1);
  std::vector<int> shortLoop = {1, 0, 3, 2};  // 0 -> 1 -> 0
  EXPECT_EQ(ncclInternalError, ncclBuildRings(1, rings.data(), 0, 4, nullptr, shortLoop.data()));
  EXPECT_TRUE(Logged("ring 0 does not contain rank 2"));

  std::vector<int> open = {1, 2, 3, 1};  // 3 goes back to 1, never to 0
  EXPECT_EQ(ncclInternalError, ncclBuildRings(1, rings.data(), 0, 4, nullptr, open.data()));
  EXPECT_TRUE(Logged("ring 0 does not loop back to start (1 != 0)"));
}

TEST_F(RingsMicrotest, BuildRings_ChecksRanksPastTheFirst64) {
  int n = 130;
  std::vector<int> next(n), rings(n);
  for (int r = 0; r < n; r++) next[r] = (r + 1) % n;
  ASSERT_EQ(ncclSuccess, ncclBuildRings(1, rings.data(), 5, n, nullptr, next.data()));
  EXPECT_EQ(5, rings[0]);
  EXPECT_EQ(4, rings[n - 1]);

  next[63] = 65;  // 0..63 and 65 loop twice in 130 steps, skipping 64
  next[65] = 0;
  EXPECT_EQ(ncclInternalError, ncclBuildRings(1, rings.data(), 0, n, nullptr, next.data()));
  EXPECT_TRUE(Logged("ring 0 does not contain rank 64"));
}

TEST_F(RingsMicrotest, BuildRings_EachRingIsCheckedOnItsOwn) {
  std::vector<int> next = {1, 2, 3, 0, 1, 0, 3, 2};  // channel 1: 0 -> 1 -> 0
  std::vector<int> rings(8, -1);

  EXPECT_EQ(ncclInternalError, ncclBuildRings(2, rings.data(), 0, 4, nullptr, next.data()));

  EXPECT_TRUE(Logged("ring 1 does not contain rank 2"));
}

TEST_F(RingsMicrotest, DumpLine_PadsToTheWidestRankAndTruncates) {
  std::vector<int> values = {0, 7, 12};
  dumpLine(values.data(), 12, "p:");
  EXPECT_TRUE(Logged("p:  0  7 12"));

  std::vector<int> many(100);
  for (int i = 0; i < 100; i++) many[i] = i;
  dumpLine(many.data(), 100, "long:");
  ASSERT_FALSE(g_logs.empty());
  const std::string& line = g_logs.back().text;
  EXPECT_EQ("...", line.substr(line.size() - 3));
  EXPECT_NE(std::string::npos, line.find("long:  0  1  2"));
  EXPECT_EQ(std::string::npos, line.find(" 99"));
}

TEST_F(RingsMicrotest, RcclBuildRings_WalksNextAndChecksPrev) {
  Links l = LinksOf({{0, 1, 2, 3}, {0, 2, 1, 3}});
  std::vector<int> rings(8, -1);

  ASSERT_EQ(ncclSuccess, rcclBuildRings(2, rings.data(), 2, 4, l.prev.data(), l.next.data()));
  EXPECT_EQ(std::vector<int>({2, 3, 0, 1, 2, 1, 3, 0}), rings);
  EXPECT_FALSE(Logged("Channel 00/02"));

  ASSERT_EQ(ncclSuccess, rcclBuildRings(2, rings.data(), 0, 4, l.prev.data(), l.next.data()));
  EXPECT_TRUE(Logged("Channel 01/02 : 0 2 1 3"));
  EXPECT_EQ(ncclSuccess, rcclBuildRings(0, rings.data(), 0, 4, l.prev.data(), l.next.data()));
}

TEST_F(RingsMicrotest, RcclBuildRings_RejectsBadArguments) {
  Links l = LinksOf({{0, 1}});
  std::vector<int> rings(2);
  int* r = rings.data();

  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(-1, r, 0, 2, l.prev.data(), l.next.data()));
  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(1, r, -1, 2, l.prev.data(), l.next.data()));
  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(1, r, 0, -1, l.prev.data(), l.next.data()));
  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(1, nullptr, 0, 2, l.prev.data(), l.next.data()));
  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(1, r, 0, 2, nullptr, l.next.data()));
  EXPECT_EQ(ncclInvalidArgument, rcclBuildRings(1, r, 0, 2, l.prev.data(), nullptr));
}

// The loop-back and completeness checks after the walk cannot fail: with prev[next[x]] == x enforced, any
// early return to a visited rank is caught as a sub-loop or an asymmetric link first.
TEST_F(RingsMicrotest, RcclBuildRings_BrokenLinksAreErrors) {
  std::vector<int> rings(4);
  Links l = LinksOf({{0, 1, 2, 3}});

  EXPECT_EQ(ncclInternalError, rcclBuildRings(1, rings.data(), 4, 4, l.prev.data(), l.next.data()));
  EXPECT_TRUE(Logged("Ring 0: Found invalid rank index 4"));

  l.next[1] = 0;
  l.prev[0] = 1;
  EXPECT_EQ(ncclInternalError, rcclBuildRings(1, rings.data(), 0, 4, l.prev.data(), l.next.data()));
  EXPECT_TRUE(Logged("Ring 0: Unexpected sub-loop detected at rank 0"));

  l = LinksOf({{0, 1, 2, 3}});
  l.prev[1] = 3;
  EXPECT_EQ(ncclInternalError, rcclBuildRings(1, rings.data(), 0, 4, l.prev.data(), l.next.data()));
  EXPECT_TRUE(Logged("Ring 0: Asymmetric link! Rank 0 -> 1, but 1 -> 3"));

  l = LinksOf({{0, 1, 2, 3}});
  l.next[2] = 7;
  EXPECT_EQ(ncclInternalError, rcclBuildRings(1, rings.data(), 0, 4, l.prev.data(), l.next.data()));
  EXPECT_TRUE(Logged("Ring 0: Rank 2 pointed to invalid next_rank 7"));
}

}  // namespace
