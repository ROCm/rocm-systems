// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_report.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace rocjitsu::consan {
namespace {

TEST(AutoReportPlan, EmptyIsExactlyOneHeader) {
  const auto plan = plan_auto_report({});
  EXPECT_TRUE(plan.complete());
  EXPECT_EQ(plan.outcome, AutoReportPlanOutcome::Complete);
  EXPECT_EQ(plan.reason, AutoReportPlanReason::None);
  EXPECT_EQ(plan.required_bytes, sizeof(ReportHeader));
  EXPECT_EQ(plan.layout.required_bytes, sizeof(ReportHeader));
  EXPECT_TRUE(plan.layout.valid);
  EXPECT_EQ(auto_report_plan_outcome_name(plan.outcome), "complete");
}

TEST(AutoReportPlan, SeparatesBanksFromMultiCellWatchpoints) {
  const AutoReportInventory inventory{.range_bank_count = 17, .watchpoint_count = 53};
  const auto plan = plan_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.causal_window_capacity, 17u);
  EXPECT_EQ(plan.layout.watchpoint_capacity, 53u);
  EXPECT_EQ(plan.layout.sync_metadata_capacity, 17u);
  EXPECT_EQ(plan.layout.pending_acquire_capacity, 17u);
  EXPECT_EQ(plan.layout.causal_windows_offset, sizeof(ReportHeader));
  EXPECT_EQ(plan.layout.watchpoints_offset,
            plan.layout.causal_windows_offset + 17 * sizeof(CausalWindow));
  EXPECT_EQ(plan.layout.sync_metadata_offset,
            plan.layout.watchpoints_offset + 53 * sizeof(uint64_t));
  EXPECT_EQ(plan.layout.pending_acquires_offset,
            plan.layout.sync_metadata_offset + 17 * sizeof(SyncMetadataPacked));
  EXPECT_EQ(plan.required_bytes,
            plan.layout.pending_acquires_offset + 17 * sizeof(PendingAcquireSlot));
}

TEST(AutoReportPlan, BoundaryIsExactAndOneWatchpointFails) {
  ASSERT_EQ((kOrdinaryAutoReportBufferCeilingBytes - sizeof(ReportHeader)) % sizeof(uint64_t), 0u);
  const uint64_t watchpoint_count =
      (kOrdinaryAutoReportBufferCeilingBytes - sizeof(ReportHeader)) / sizeof(uint64_t);
  AutoReportInventory inventory{.watchpoint_count = watchpoint_count};
  const auto accepted = plan_auto_report(inventory);
  ASSERT_TRUE(accepted.complete());
  EXPECT_EQ(accepted.required_bytes, kOrdinaryAutoReportBufferCeilingBytes);

  ++inventory.watchpoint_count;
  const auto rejected = plan_auto_report(inventory);
  EXPECT_EQ(rejected.outcome, AutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(rejected.reason, AutoReportPlanReason::PerBufferCeiling);
  EXPECT_EQ(rejected.required_bytes, kOrdinaryAutoReportBufferCeilingBytes + sizeof(uint64_t));
}

TEST(AutoReportPlan, AdaptiveBanksFitWithoutDroppingLogicalRanges) {
  constexpr uint64_t kLogicalRanges = 135610u;
  const AutoReportInventory requested{
      .access_range_count = kLogicalRanges,

      .range_bank_count = 8u * kLogicalRanges,
      .watchpoint_count = 8u * kLogicalRanges,
      .bank_count_adaptive = true,
  };
  ASSERT_EQ(plan_auto_report(requested).outcome, AutoReportPlanOutcome::InsufficientReportCapacity);

  const auto fitted = fit_auto_report_inventory(requested);
  EXPECT_EQ(fitted.access_range_count, kLogicalRanges);
  EXPECT_EQ(fitted.range_bank_count, 4u * kLogicalRanges);
  EXPECT_EQ(fitted.watchpoint_count, 4u * kLogicalRanges);
  EXPECT_TRUE(plan_auto_report(fitted).complete());

  auto exact = requested;
  exact.bank_count_adaptive = false;
  EXPECT_EQ(fit_auto_report_inventory(exact).range_bank_count, 8u * kLogicalRanges);
}

TEST(AutoReportPlan, AdaptiveBanksHonorExplicitCallerCap) {
  constexpr uint64_t kCallerCap = 1024u * 1024u;
  constexpr uint64_t kLogicalRanges = 4096u;
  const AutoReportInventory requested{
      .access_range_count = kLogicalRanges,

      .range_bank_count = 8u * kLogicalRanges,
      .watchpoint_count = 8u * kLogicalRanges,
      .bank_count_adaptive = true,
  };
  ASSERT_TRUE(plan_auto_report(requested).complete());
  ASSERT_EQ(plan_auto_report(requested, kCallerCap).outcome,
            AutoReportPlanOutcome::InsufficientReportCapacity);

  const auto fitted = fit_auto_report_inventory(requested, kCallerCap);
  EXPECT_EQ(fitted.access_range_count, kLogicalRanges);
  EXPECT_LT(fitted.range_bank_count, requested.range_bank_count);
  EXPECT_GE(fitted.range_bank_count, kLogicalRanges);
  const auto accepted = plan_auto_report(fitted, kCallerCap);
  ASSERT_TRUE(accepted.complete());
  EXPECT_LE(accepted.required_bytes, kCallerCap);
}

TEST(AutoReportPlan, EveryAbiCapacityRejectsOnePastUint32) {
  constexpr uint64_t overflow = uint64_t{std::numeric_limits<uint32_t>::max()} + 1u;
  const AutoReportInventory inventories[] = {
      {.range_bank_count = overflow},
      {.sync_slot_count = overflow},
      {.watchpoint_count = overflow},
  };
  for (const auto &inventory : inventories) {
    const auto plan = plan_auto_report(inventory);
    EXPECT_EQ(plan.outcome, AutoReportPlanOutcome::Overflow);
    EXPECT_EQ(plan.reason, AutoReportPlanReason::AbiCapacityOverflow);
    EXPECT_EQ(plan.required_bytes, 0u);
    EXPECT_FALSE(plan.layout.valid);
  }
}

TEST(AutoReportPlan, RepresentableHugeCountsAreCapacityInsufficientNotOverflow) {
  const AutoReportInventory inventory{
      .range_bank_count = std::numeric_limits<uint32_t>::max(),
      .watchpoint_count = std::numeric_limits<uint32_t>::max(),
  };
  const auto plan = plan_auto_report(inventory);
  EXPECT_EQ(plan.outcome, AutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(plan.reason, AutoReportPlanReason::PerBufferCeiling);
  EXPECT_GT(plan.required_bytes, kOrdinaryAutoReportBufferCeilingBytes);
  EXPECT_FALSE(plan.layout.valid);
}

TEST(AutoReportPlan, FrozenSafetyCeilingsRemainDistinct) {
  EXPECT_EQ(kOrdinaryAutoReportBufferCeilingBytes, 128u * 1024u * 1024u);
  EXPECT_EQ(kAutoReportProcessCeilingBytes, 4ull * 1024u * 1024u * 1024u);
  EXPECT_GT(kAutoReportProcessCeilingBytes, kOrdinaryAutoReportBufferCeilingBytes);
  EXPECT_EQ(auto_report_plan_reason_name(AutoReportPlanReason::PerBufferCeiling),
            "per_buffer_ceiling");
}

TEST(AutoReportPlan, ProcessBudgetIsInclusiveAcrossObjectsAndReleaseIsChecked) {
  AutoReportProcessBudget budget;
  constexpr uint64_t first = 16u * 1024u * 1024u;
  ASSERT_TRUE(reserve_auto_report_bytes(budget, first));
  ASSERT_TRUE(reserve_auto_report_bytes(budget, kAutoReportProcessCeilingBytes - first));
  EXPECT_EQ(budget.current_live_bytes, kAutoReportProcessCeilingBytes);
  EXPECT_EQ(budget.peak_live_bytes, kAutoReportProcessCeilingBytes);
  EXPECT_FALSE(reserve_auto_report_bytes(budget, 1u));
  EXPECT_FALSE(release_auto_report_bytes(budget, kAutoReportProcessCeilingBytes + 1u));
  ASSERT_TRUE(release_auto_report_bytes(budget, first));
  ASSERT_TRUE(release_auto_report_bytes(budget, kAutoReportProcessCeilingBytes - first));
  EXPECT_EQ(budget.current_live_bytes, 0u);
  EXPECT_EQ(budget.peak_live_bytes, kAutoReportProcessCeilingBytes);
}

TEST(AutoReportPlan, ProcessBudgetAdmitsMaximumReports) {
  AutoReportProcessBudget budget;
  for (unsigned object = 0;
       object < kAutoReportProcessCeilingBytes / kOrdinaryAutoReportBufferCeilingBytes; ++object)
    ASSERT_TRUE(reserve_auto_report_bytes(budget, kOrdinaryAutoReportBufferCeilingBytes));
  EXPECT_EQ(budget.current_live_bytes, kAutoReportProcessCeilingBytes);
  EXPECT_FALSE(reserve_auto_report_bytes(budget, 1u));
}

TEST(AutoReportPlan, CanonicalLayoutRoundTripsHeterogeneousLayout) {
  const AutoReportInventory inventory{.range_bank_count = 5, .watchpoint_count = 17};
  const AutoReportPlan plan = plan_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  const auto candidate_layout = plan.complete_layout();
  ASSERT_TRUE(candidate_layout);
  EXPECT_EQ(candidate_layout->causal_window_capacity, 5u);
  EXPECT_EQ(candidate_layout->watchpoint_capacity, 17u);

  const ReportBufferLayout resolved =
      revalidate_report_layout(*candidate_layout, plan.required_bytes);
  EXPECT_TRUE(resolved.valid);
  EXPECT_EQ(resolved.required_bytes, plan.layout.required_bytes);
  EXPECT_EQ(resolved.causal_windows_offset, plan.layout.causal_windows_offset);
  EXPECT_EQ(resolved.watchpoints_offset, plan.layout.watchpoints_offset);
  EXPECT_EQ(resolved.pending_acquires_offset, plan.layout.pending_acquires_offset);
  const ReportHeader header = make_report_header_for_layout(
      /*generation=*/7, /*dispatch_id=*/9, resolved);
  EXPECT_EQ(header.watchpoint_capacity, 17u);
  EXPECT_EQ(header.causal_window_capacity, 5u);
  EXPECT_EQ(header.sync_metadata_capacity, 5u);
  EXPECT_EQ(header.pending_acquire_capacity, 5u);
  EXPECT_TRUE(report_layout_matches_header(header, resolved, plan.required_bytes));
}

TEST(AutoReportPlan, CanonicalLayoutRejectsCorruptOffsetAndShortAllocation) {
  const AutoReportPlan plan =
      plan_auto_report({.range_bank_count = 2, .sync_slot_count = 3, .watchpoint_count = 4});
  ASSERT_TRUE(plan.complete());
  auto candidate_layout = plan.complete_layout();
  ASSERT_TRUE(candidate_layout);

  ++candidate_layout->sync_metadata_offset;
  EXPECT_FALSE(revalidate_report_layout(*candidate_layout, plan.required_bytes).valid);
  --candidate_layout->sync_metadata_offset;
  EXPECT_FALSE(revalidate_report_layout(*candidate_layout, plan.required_bytes - 1u).valid);
}

TEST(AutoReportPlan, IncompletePlanCannotProduceACompleteLayout) {
  const AutoReportPlan plan =
      plan_auto_report({.range_bank_count = kOrdinaryAutoReportBufferCeilingBytes});
  ASSERT_FALSE(plan.complete());
  EXPECT_FALSE(plan.complete_layout());
}

TEST(AutoReportPlan, CompleteLayoutRequiresValidConsistentGeometry) {
  AutoReportPlan plan = plan_auto_report({.range_bank_count = 2, .watchpoint_count = 3});
  ASSERT_TRUE(plan.complete_layout());

  plan.layout.valid = false;
  EXPECT_FALSE(plan.complete_layout());
  plan.layout.valid = true;
  ++plan.layout.required_bytes;
  EXPECT_FALSE(plan.complete_layout());
}

} // namespace
} // namespace rocjitsu::consan
