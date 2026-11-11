// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simdojo_issue_queue_test.cpp
/// @brief Tests for simdojo::IssueQueue: ready cycles and in-order departure.

#include "simdojo/components/issue_queue.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace simdojo;

namespace {

TEST(IssueQueueTest, EntryLeavesFromItsReadyCycle) {
  IssueQueue<int> queue;
  EXPECT_TRUE(queue.empty());
  queue.push(7, 4);
  EXPECT_FALSE(queue.empty());
  EXPECT_FALSE(queue.ready(3));
  ASSERT_TRUE(queue.ready(4));
  EXPECT_TRUE(queue.ready(5));
  EXPECT_EQ(queue.pop(), 7);
  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.ready(5));
}

TEST(IssueQueueTest, FixedLatencySustainsOneEntryPerCycle) {
  constexpr uint64_t kLatency = 3;
  IssueQueue<int> queue;
  std::vector<std::pair<uint64_t, int>> left;
  for (uint64_t cycle = 0; cycle < 8; ++cycle) {
    if (queue.ready(cycle))
      left.emplace_back(cycle, queue.pop());
    queue.push(static_cast<int>(cycle), cycle + kLatency);
  }
  const std::vector<std::pair<uint64_t, int>> expected{{3, 0}, {4, 1}, {5, 2}, {6, 3}, {7, 4}};
  EXPECT_EQ(left, expected);
}

TEST(IssueQueueTest, YoungerEntryWaitsBehindAnOlderOne) {
  IssueQueue<int> queue;
  queue.push(1, 10);
  queue.push(2, 5);
  EXPECT_FALSE(queue.ready(5));
  ASSERT_TRUE(queue.ready(10));
  EXPECT_EQ(queue.pop(), 1);
  ASSERT_TRUE(queue.ready(10));
  EXPECT_EQ(queue.pop(), 2);
}

TEST(IssueQueueTest, CarriesMoveOnlyEntries) {
  IssueQueue<std::unique_ptr<int>> queue;
  queue.push(std::make_unique<int>(3), 1);
  ASSERT_TRUE(queue.ready(1));
  auto value = queue.pop();
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 3);
}

} // namespace
