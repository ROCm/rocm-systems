// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"

#include "util/bit.h"

#include <algorithm>

namespace rocjitsu::consan_moi_impl {

bool plan_sampled_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  const uint64_t sampled_sync_slot_count =
      std::max(inventory.sampled_range_bank_count, inventory.sampled_sync_slot_count);
  // Access-only objects never create deferred acquires. Once an atomic is
  // admitted, each causal slot must retain every wave's acquire until the
  // associated later access executes.
  const uint64_t pending_owner_bank_count =
      inventory.atomic_event_count == 0u ? 1u : kConSanMoiSampledPendingAcquireOwnerBankCount;
  const auto pending_acquire_count =
      util::checked_mul(sampled_sync_slot_count, pending_owner_bank_count);
  if (!pending_acquire_count) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return plan_moi_report_regions(
      {moi_report_region<ConSanMoiDiagnosticRecord>(inventory.diagnostic_count,
                                                    layout.diagnostic_capacity,
                                                    layout.diagnostic_records_offset),
       moi_report_region<ConSanMoiSampledCausalWindow>(sampled_sync_slot_count,
                                                       layout.sampled_causal_window_capacity,
                                                       layout.sampled_causal_windows_offset),
       moi_report_region<uint64_t>(inventory.sampled_watchpoint_count,
                                   layout.sampled_watchpoint_capacity,
                                   layout.sampled_watchpoints_offset),
       moi_report_region<ConSanMoiSampledSyncMetadataPacked>(sampled_sync_slot_count,
                                                             layout.sampled_sync_metadata_capacity,
                                                             layout.sampled_sync_metadata_offset),
       moi_report_region<ConSanMoiSampledPendingAcquireSlot>(
           *pending_acquire_count, layout.sampled_pending_acquire_capacity,
           layout.sampled_pending_acquires_offset)},
      plan, cursor);
}

std::optional<ConSanMoiAutoReportInventory>
reconstruct_sampled_report_inventory(const ConSanMoiReportBufferLayout &candidate) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::Sampled;
  inventory.diagnostic_count = candidate.diagnostic_capacity;
  inventory.sampled_range_bank_count = candidate.sampled_causal_window_capacity;
  inventory.sampled_sync_slot_count = candidate.sampled_causal_window_capacity;
  inventory.sampled_watchpoint_count = candidate.sampled_watchpoint_capacity;
  if (candidate.sampled_causal_window_capacity != 0u &&
      candidate.sampled_pending_acquire_capacity / candidate.sampled_causal_window_capacity ==
          kConSanMoiSampledPendingAcquireOwnerBankCount &&
      candidate.sampled_pending_acquire_capacity % candidate.sampled_causal_window_capacity == 0u) {
    // Reconstruct the atomic-present layout class from the retained
    // capacities so exact auto layouts round-trip through the planner.
    inventory.atomic_event_count = 1u;
  }
  return inventory;
}

} // namespace rocjitsu::consan_moi_impl

namespace rocjitsu {

ConSanMoiAutoReportInventory
fit_consan_moi_sampled_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                             uint64_t caller_ceiling_bytes) {
  if (inventory.engine != ConSanMoiEngine::Sampled || !inventory.sampled_bank_count_adaptive ||
      inventory.access_range_count == 0u ||
      inventory.sampled_watchpoint_count < inventory.sampled_range_bank_count ||
      inventory.sampled_range_bank_count % inventory.access_range_count != 0u) {
    return inventory;
  }

  const uint64_t reserved_sync_slots =
      std::max(inventory.sampled_sync_slot_count, inventory.sampled_range_bank_count) -
      inventory.sampled_range_bank_count;
  const uint64_t extra_watchpoints =
      inventory.sampled_watchpoint_count - inventory.sampled_range_bank_count;
  uint64_t bank_count = inventory.sampled_range_bank_count / inventory.access_range_count;
  if (bank_count == 0u || bank_count > 8u || (bank_count & (bank_count - 1u)) != 0u)
    return inventory;

  while (bank_count > 1u) {
    const ConSanMoiAutoReportPlan plan =
        plan_consan_moi_auto_report(inventory, caller_ceiling_bytes);
    if (plan.complete() ||
        plan.outcome != ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity ||
        plan.reason != ConSanMoiAutoReportPlanReason::PerBufferCeiling) {
      return inventory;
    }
    bank_count /= 2u;
    inventory.sampled_range_bank_count = inventory.access_range_count * bank_count;
    inventory.sampled_sync_slot_count =
        util::saturating_add(inventory.sampled_range_bank_count, reserved_sync_slots);
    inventory.sampled_watchpoint_count =
        util::saturating_add(inventory.sampled_range_bank_count, extra_watchpoints);
  }
  return inventory;
}

} // namespace rocjitsu
