// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/aligned_atomic.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace {

static_assert(alignof(util::AlignedAtomic<int>) == 64);
static_assert(sizeof(util::AlignedAtomic<int>) == 64);
static_assert(!std::is_copy_constructible_v<util::AlignedAtomic<int>>);
static_assert(!std::is_copy_assignable_v<util::AlignedAtomic<int>>);
static_assert(!std::is_move_constructible_v<util::AlignedAtomic<int>>);

struct alignas(128) WideValue {
  uint64_t value;
};
static_assert(alignof(util::AlignedAtomic<WideValue>) == alignof(std::atomic<WideValue>));

TEST(AlignedAtomic, PreservesInitializationAssignmentAndBaseOperations) {
  util::AlignedAtomic<int> empty;
  EXPECT_EQ(empty.load(), 0);
  util::AlignedAtomic<int> value{7};
  EXPECT_EQ(value.load(), 7);
  value = 11;
  std::atomic<int> &base = value;
  EXPECT_EQ(base.fetch_add(3), 11);
  int expected = 14;
  EXPECT_TRUE(value.compare_exchange_strong(expected, 19));
  EXPECT_EQ(value.load(), 19);
}

TEST(AlignedAtomic, PreservesBooleanOperations) {
  util::AlignedAtomic<bool> flag{false};
  flag = true;
  EXPECT_TRUE(flag.exchange(false));
  EXPECT_FALSE(flag.load());
}

TEST(AlignedAtomic, PreservesPointerArithmetic) {
  int values[3]{};
  util::AlignedAtomic<int *> pointer{values};
  EXPECT_EQ(pointer.fetch_add(1), values);
  EXPECT_EQ(pointer.load(), values + 1);
  pointer = values + 2;
  EXPECT_EQ(pointer.fetch_sub(1), values + 2);
  EXPECT_EQ(pointer.load(), values + 1);
}

TEST(AlignedAtomic, SupportsSharedOwnershipThroughAtomicBase) {
  std::shared_ptr<std::atomic<uint64_t>> token;
  {
    auto aligned = std::make_shared<util::AlignedAtomic<uint64_t>>(1);
    token = aligned;
    EXPECT_EQ(token->fetch_add(1), 1u);
    EXPECT_EQ(aligned->load(), 2u);
  }
  EXPECT_EQ(token->load(), 2u);
}

} // namespace
