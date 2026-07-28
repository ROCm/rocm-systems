// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file checked_byte_budget_test.cpp
/// @brief Unit tests for shared checked byte arithmetic and fixed budgets.

#include "rocjitsu/checked_byte_budget.h"
#include "util/bit.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu {
namespace {

TEST(CheckedByteBudgetTest, AllocationChargeHandlesExactFitsAndOverflow) {
  using byte_accounting::checked_allocation_charge;

  EXPECT_EQ(checked_allocation_charge(5, 3, 7), std::optional<uint64_t>(26));
  EXPECT_FALSE(checked_allocation_charge(std::numeric_limits<uint64_t>::max(), 1, 1));
  EXPECT_FALSE(checked_allocation_charge(0, std::numeric_limits<uint64_t>::max(), 2));
}

TEST(CheckedByteBudgetTest, ExcessCapacityChargeHandlesCapacityBoundaries) {
  using byte_accounting::excess_capacity_charge;

  EXPECT_EQ(excess_capacity_charge(12, 8, 10), std::optional<uint64_t>(40));
  EXPECT_FALSE(excess_capacity_charge(7, 8, 10));
  EXPECT_FALSE(excess_capacity_charge(std::numeric_limits<uint64_t>::max(), 0, 2));
}

TEST(CheckedByteBudgetTest, CheckedAlignmentHandlesArbitraryAlignmentAndOverflow) {
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();

  EXPECT_EQ(util::checked_align_up(16u, 8u), std::optional<unsigned>(16u));
  EXPECT_EQ(util::checked_align_up(10u, 3u), std::optional<unsigned>(12u));
  EXPECT_EQ(util::checked_align_up(0u, 3u), std::optional<unsigned>(0u));
  EXPECT_FALSE(util::checked_align_up(17u, 0u));
  EXPECT_EQ(util::checked_align_up(maximum - 7u, uint64_t{8}),
            std::optional<uint64_t>(maximum - 7u));
  EXPECT_FALSE(util::checked_align_up(maximum - 6u, uint64_t{8}));
}

TEST(CheckedByteBudgetTest, DistinguishesFailuresWithoutMutatingUsedBytes) {
  using byte_accounting::ChargeOutcome;

  byte_accounting::CheckedByteBudget budget(26);
  const auto first = budget.charge(5);
  ASSERT_TRUE(first);
  EXPECT_EQ(first.outcome, ChargeOutcome::WithinLimit);
  EXPECT_EQ(first.previous_used_bytes, 0u);
  EXPECT_EQ(first.required_bytes, 5u);

  const auto limit = budget.charge(22);
  EXPECT_EQ(limit.outcome, ChargeOutcome::LimitExceeded);
  EXPECT_EQ(limit.previous_used_bytes, 5u);
  EXPECT_EQ(limit.required_bytes, 27u);
  EXPECT_EQ(limit.limit_bytes, 26u);
  EXPECT_EQ(budget.used_bytes(), 5u);

  const auto overflow = budget.charge_allocation(std::numeric_limits<uint64_t>::max(), 2);
  EXPECT_EQ(overflow.outcome, ChargeOutcome::AccountingOverflow);
  EXPECT_FALSE(overflow.required_bytes);
  EXPECT_EQ(budget.used_bytes(), 5u);

  EXPECT_TRUE(budget.charge_allocation(3, 7));
  EXPECT_EQ(budget.used_bytes(), 26u);
}

TEST(CheckedByteBudgetTest, SaturatingMultiplicationPinsOverflow) {
  using byte_accounting::saturating_multiply;

  EXPECT_EQ(saturating_multiply(3, 7), 21u);
  EXPECT_EQ(saturating_multiply(0, std::numeric_limits<uint64_t>::max()), 0u);
  EXPECT_EQ(saturating_multiply(2, std::numeric_limits<uint64_t>::max()),
            std::numeric_limits<uint64_t>::max());
}

} // namespace
} // namespace rocjitsu
