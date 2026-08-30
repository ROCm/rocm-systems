// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"

#include "util/bit.h"

#include <algorithm>
#include <limits>

namespace rocjitsu::consan_moi_impl {

bool plan_inline_shadow_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                      ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  // Provision one external slot per LDS byte. Objects containing subword
  // traffic use those slots directly; four-byte cells use a bounded subset.
  const uint64_t exact_shadow_cells = inventory.inline_lds_bytes;
  if (exact_shadow_cells > std::numeric_limits<uint32_t>::max()) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  layout.inline_exact_dispatch_bank_count =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes);
  if (layout.inline_exact_dispatch_bank_count == 0u) {
    plan.outcome = ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity;
    plan.reason = ConSanMoiAutoReportPlanReason::PerBufferCeiling;
    return false;
  }
  const auto exact_shadow_count = util::checked_mul(
      exact_shadow_cells, static_cast<uint64_t>(layout.inline_exact_dispatch_bank_count));
  if (!exact_shadow_count) {
    plan.reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  if (!checked_moi_report_capacity(inventory.diagnostic_count, layout.diagnostic_capacity) ||
      !checked_moi_report_capacity(*exact_shadow_count, layout.exact_shadow_entry_capacity) ||
      !checked_moi_report_capacity(inventory.inline_atomic_release_count,
                                   layout.inline_atomic_release_capacity) ||
      !checked_moi_report_capacity(inventory.inline_causal_snapshot_count,
                                   layout.inline_causal_snapshot_capacity) ||
      !checked_moi_report_capacity(inventory.inline_compact_token_mapping_count,
                                   layout.inline_compact_token_mapping_capacity) ||
      !checked_moi_report_capacity(inventory.inline_acquired_epoch_token_count,
                                   layout.inline_acquired_epoch_token_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_moi_report_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                                  alignof(ConSanMoiDiagnosticRecord), cursor,
                                  layout.diagnostic_records_offset) &&
         append_moi_report_region(*exact_shadow_count, sizeof(ConSanMoiInlineExactShadowSlot),
                                  alignof(ConSanMoiInlineExactShadowSlot), cursor,
                                  layout.exact_shadow_entries_offset) &&
         append_moi_report_region(inventory.inline_atomic_release_count,
                                  sizeof(ConSanMoiInlineAtomicReleaseSlot),
                                  alignof(ConSanMoiInlineAtomicReleaseSlot), cursor,
                                  layout.inline_atomic_release_slots_offset) &&
         append_moi_report_region(inventory.inline_causal_snapshot_count,
                                  sizeof(ConSanMoiInlineCausalSnapshot),
                                  alignof(ConSanMoiInlineCausalSnapshot), cursor,
                                  layout.inline_causal_snapshots_offset) &&
         append_moi_report_region(inventory.inline_compact_token_mapping_count,
                                  sizeof(ConSanMoiCompactDiagnosticTokenMapping),
                                  alignof(ConSanMoiCompactDiagnosticTokenMapping), cursor,
                                  layout.inline_compact_token_mappings_offset) &&
         append_moi_report_region(inventory.inline_acquired_epoch_token_count,
                                  sizeof(ConSanMoiInlineAcquiredEpochTokenSlot),
                                  alignof(ConSanMoiInlineAcquiredEpochTokenSlot), cursor,
                                  layout.inline_acquired_epoch_token_slots_offset);
}

std::optional<ConSanMoiAutoReportInventory>
reconstruct_inline_shadow_report_inventory(const ConSanMoiReportBufferLayout &candidate) {
  if (candidate.inline_exact_dispatch_bank_count == 0u ||
      candidate.inline_exact_dispatch_bank_count > kConSanMoiInlineMaximumDispatchBankCount ||
      (candidate.inline_exact_dispatch_bank_count &
       (candidate.inline_exact_dispatch_bank_count - 1u)) != 0u ||
      candidate.exact_shadow_entry_capacity % candidate.inline_exact_dispatch_bank_count) {
    return std::nullopt;
  }
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::InlineShadow;
  inventory.diagnostic_count = candidate.diagnostic_capacity;
  inventory.inline_lds_bytes = static_cast<uint64_t>(candidate.exact_shadow_entry_capacity) /
                               candidate.inline_exact_dispatch_bank_count;
  inventory.inline_atomic_release_count = candidate.inline_atomic_release_capacity;
  inventory.inline_causal_snapshot_count = candidate.inline_causal_snapshot_capacity;
  inventory.inline_compact_token_mapping_count = candidate.inline_compact_token_mapping_capacity;
  inventory.inline_acquired_epoch_token_count = candidate.inline_acquired_epoch_token_capacity;
  return inventory;
}

} // namespace rocjitsu::consan_moi_impl

namespace rocjitsu {

ConSanMoiAutoReportInventory
fit_consan_moi_inline_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                            uint64_t caller_ceiling_bytes) {
  if (inventory.engine != ConSanMoiEngine::InlineShadow ||
      !inventory.inline_diagnostic_count_adaptive) {
    return inventory;
  }

  const ConSanMoiAutoReportPlan requested =
      plan_consan_moi_auto_report(inventory, caller_ceiling_bytes);
  if (requested.complete() ||
      requested.outcome != ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity ||
      requested.reason != ConSanMoiAutoReportPlanReason::PerBufferCeiling) {
    return inventory;
  }

  const uint64_t minimum = std::min<uint64_t>(
      inventory.diagnostic_count, std::max(kConSanMoiInlineShadowDefaultDiagnosticCapacity,
                                           kConSanMoiInlineShadowDiagnosticHeadroomPerAccess));
  ConSanMoiAutoReportInventory candidate = inventory;
  candidate.diagnostic_count = minimum;
  if (!plan_consan_moi_auto_report(candidate, caller_ceiling_bytes).complete())
    return candidate;

  uint64_t fitting = minimum;
  uint64_t rejected = inventory.diagnostic_count;
  while (fitting < rejected) {
    const uint64_t midpoint = fitting + (rejected - fitting + 1u) / 2u;
    candidate.diagnostic_count = midpoint;
    if (plan_consan_moi_auto_report(candidate, caller_ceiling_bytes).complete())
      fitting = midpoint;
    else
      rejected = midpoint - 1u;
  }
  inventory.diagnostic_count = fitting;
  return inventory;
}

} // namespace rocjitsu
