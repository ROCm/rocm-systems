// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/inline_vector.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

struct CountingValue {
  static inline int live = 0;
  static inline int constructed = 0;
  static inline int destroyed = 0;

  int value = 0;

  CountingValue() noexcept { count_construct(); }
  explicit CountingValue(int v) noexcept : value(v) { count_construct(); }

  CountingValue(const CountingValue &other) noexcept : value(other.value) { count_construct(); }

  CountingValue(CountingValue &&other) noexcept : value(other.value) {
    other.value = -1;
    count_construct();
  }

  CountingValue &operator=(const CountingValue &other) noexcept {
    value = other.value;
    return *this;
  }

  CountingValue &operator=(CountingValue &&other) noexcept {
    value = other.value;
    other.value = -1;
    return *this;
  }

  ~CountingValue() noexcept {
    --live;
    ++destroyed;
  }

  static void reset() noexcept {
    live = 0;
    constructed = 0;
    destroyed = 0;
  }

private:
  static void count_construct() noexcept {
    ++live;
    ++constructed;
  }
};

TEST(InlineVectorTest, ZeroInlineCapacityUsesCompactDynamicLayout) {
  using DynamicVector = util::inline_vector<int, 0>;

  EXPECT_EQ(sizeof(DynamicVector), sizeof(void *) + 2 * sizeof(uint32_t));

  DynamicVector values;
  EXPECT_TRUE(values.empty());
  EXPECT_EQ(values.capacity(), 0u);
  EXPECT_TRUE(values.is_inline());

  values.push_back(7);
  EXPECT_EQ(values.size(), 1u);
  EXPECT_GE(values.capacity(), 1u);
  EXPECT_FALSE(values.is_inline());
  EXPECT_EQ(values[0], 7);
}

TEST(InlineVectorTest, KeepsSmallValuesInlineThenGrows) {
  util::inline_vector<int, 4> values;
  int *inline_data = values.data();

  values.push_back(1);
  values.push_back(2);
  values.push_back(3);
  values.push_back(4);

  EXPECT_TRUE(values.is_inline());
  EXPECT_EQ(values.data(), inline_data);

  values.push_back(5);

  EXPECT_FALSE(values.is_inline());
  EXPECT_NE(values.data(), inline_data);
  ASSERT_EQ(values.size(), 5u);
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], static_cast<int>(i + 1));
  }
}

TEST(InlineVectorTest, ResizeAndEraseMaintainContiguousContents) {
  util::inline_vector<int, 2> values{1, 2, 3, 4, 5};

  auto it = values.erase(values.begin() + 1, values.begin() + 3);

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(it, values.begin() + 1);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 4);
  EXPECT_EQ(values[2], 5);

  values.resize(5, 9);

  ASSERT_EQ(values.size(), 5u);
  EXPECT_EQ(values[3], 9);
  EXPECT_EQ(values[4], 9);

  values.resize(2);

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 4);
}

TEST(InlineVectorTest, SupportsRangeConstructionAssignAndInsert) {
  const std::vector<int> source{1, 2, 5};
  util::inline_vector<int, 2> values(source.begin(), source.end());

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 5);

  values.insert(values.begin() + 2, {3, 4});

  ASSERT_EQ(values.size(), 5u);
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], static_cast<int>(i + 1));
  }

  values.assign(3, 9);

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 9);
  EXPECT_EQ(values[1], 9);
  EXPECT_EQ(values[2], 9);
}

TEST(InlineVectorTest, TracksNonTrivialElementLifetimeAcrossGrowthAndCopy) {
  CountingValue::reset();

  {
    util::inline_vector<CountingValue, 2> values;
    values.emplace_back(1);
    values.emplace_back(2);
    values.emplace_back(3);

    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(CountingValue::live, 3);

    util::inline_vector<CountingValue, 2> copy = values;

    ASSERT_EQ(copy.size(), 3u);
    EXPECT_EQ(copy[0].value, 1);
    EXPECT_EQ(copy[1].value, 2);
    EXPECT_EQ(copy[2].value, 3);
    EXPECT_EQ(CountingValue::live, 6);

    util::inline_vector<CountingValue, 2> moved = std::move(values);

    ASSERT_EQ(moved.size(), 3u);
    EXPECT_TRUE(values.empty());
    EXPECT_EQ(moved[0].value, 1);
    EXPECT_EQ(moved[1].value, 2);
    EXPECT_EQ(moved[2].value, 3);
  }

  EXPECT_EQ(CountingValue::live, 0);
  EXPECT_EQ(CountingValue::constructed, CountingValue::destroyed);
}

TEST(InlineVectorTest, SupportsMoveOnlyElements) {
  util::inline_vector<std::unique_ptr<int>, 1> values;
  values.emplace_back(std::make_unique<int>(11));
  values.emplace_back(std::make_unique<int>(22));

  util::inline_vector<std::unique_ptr<int>, 1> moved = std::move(values);

  ASSERT_EQ(moved.size(), 2u);
  EXPECT_TRUE(values.empty());
  ASSERT_NE(moved[0], nullptr);
  ASSERT_NE(moved[1], nullptr);
  EXPECT_EQ(*moved[0], 11);
  EXPECT_EQ(*moved[1], 22);
}

} // namespace
