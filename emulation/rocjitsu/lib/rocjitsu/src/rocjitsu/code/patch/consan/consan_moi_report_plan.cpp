// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "util/bit.h"

#include <bit>
#include <limits>

namespace rocjitsu {
namespace {

[[nodiscard]] bool checked_capacity(uint64_t count, uint32_t &capacity) {
  if (count > std::numeric_limits<uint32_t>::max())
    return false;
  capacity = static_cast<uint32_t>(count);
  return true;
}

[[nodiscard]] bool append_region(uint64_t count, uint64_t element_size, uint64_t alignment,
                                 uint64_t &cursor, size_t &offset) {
  if (!std::has_single_bit(alignment))
    return false;
  const auto aligned = util::checked_align_up(cursor, alignment);
  const auto end = aligned
                       ? byte_accounting::checked_allocation_charge(*aligned, count, element_size)
                       : std::nullopt;
  if (!end || *aligned > std::numeric_limits<size_t>::max())
    return false;
  offset = static_cast<size_t>(*aligned);
  cursor = *end;
  return true;
}

void alias_unused_regions(ConSanMoiReportBufferLayout &layout, size_t offset) {
  layout.record_replay_dispatch_tokens_offset = offset;
  layout.access_records_offset = offset;
  layout.barrier_records_offset = offset;
  layout.atomic_records_offset = offset;
  layout.fence_records_offset = offset;
  layout.diagnostic_records_offset = offset;
  layout.exact_shadow_entries_offset = offset;
  layout.inline_atomic_release_slots_offset = offset;
  layout.inline_acquired_epoch_token_slots_offset = offset;
  layout.inline_causal_snapshots_offset = offset;
  layout.inline_compact_token_mappings_offset = offset;
  layout.sampled_watchpoints_offset = offset;
  layout.sampled_causal_windows_offset = offset;
  layout.sampled_sync_metadata_offset = offset;
  layout.sampled_pending_acquires_offset = offset;
}

[[nodiscard]] ConSanMoiReportLayoutOverride
make_layout_override(ConSanMoiEngine engine, const ConSanMoiReportBufferLayout &layout) {
  return {
      .engine = engine,
      .access_record_capacity = layout.access_record_capacity,
      .record_replay_logical_access_range_count = layout.record_replay_logical_access_range_count,
      .record_replay_dispatch_token_capacity = layout.record_replay_dispatch_token_capacity,
      .record_replay_access_dispatch_bank_count = layout.record_replay_access_dispatch_bank_count,
      .record_replay_access_owner_bank_count = layout.record_replay_access_owner_bank_count,
      .record_replay_address_group_headroom = layout.record_replay_address_group_headroom,
      .barrier_record_capacity = layout.barrier_record_capacity,
      .atomic_record_capacity = layout.atomic_record_capacity,
      .fence_record_capacity = layout.fence_record_capacity,
      .diagnostic_capacity = layout.diagnostic_capacity,
      .exact_shadow_entry_capacity = layout.exact_shadow_entry_capacity,
      .inline_exact_dispatch_bank_count = layout.inline_exact_dispatch_bank_count,
      .inline_atomic_release_capacity = layout.inline_atomic_release_capacity,
      .inline_acquired_epoch_token_capacity = layout.inline_acquired_epoch_token_capacity,
      .inline_causal_snapshot_capacity = layout.inline_causal_snapshot_capacity,
      .inline_compact_token_mapping_capacity = layout.inline_compact_token_mapping_capacity,
      .sampled_watchpoint_capacity = layout.sampled_watchpoint_capacity,
      .sampled_causal_window_capacity = layout.sampled_causal_window_capacity,
      .sampled_sync_metadata_capacity = layout.sampled_sync_metadata_capacity,
      .sampled_pending_acquire_capacity = layout.sampled_pending_acquire_capacity,
      .record_replay_dispatch_tokens_offset = layout.record_replay_dispatch_tokens_offset,
      .access_records_offset = layout.access_records_offset,
      .barrier_records_offset = layout.barrier_records_offset,
      .atomic_records_offset = layout.atomic_records_offset,
      .fence_records_offset = layout.fence_records_offset,
      .diagnostic_records_offset = layout.diagnostic_records_offset,
      .exact_shadow_entries_offset = layout.exact_shadow_entries_offset,
      .inline_atomic_release_slots_offset = layout.inline_atomic_release_slots_offset,
      .inline_acquired_epoch_token_slots_offset = layout.inline_acquired_epoch_token_slots_offset,
      .inline_causal_snapshots_offset = layout.inline_causal_snapshots_offset,
      .inline_compact_token_mappings_offset = layout.inline_compact_token_mappings_offset,
      .sampled_watchpoints_offset = layout.sampled_watchpoints_offset,
      .sampled_causal_windows_offset = layout.sampled_causal_windows_offset,
      .sampled_sync_metadata_offset = layout.sampled_sync_metadata_offset,
      .sampled_pending_acquires_offset = layout.sampled_pending_acquires_offset,
      .required_bytes = layout.required_bytes};
}

[[nodiscard]] bool layout_override_matches(const ConSanMoiReportLayoutOverride &override_layout,
                                           const ConSanMoiReportBufferLayout &layout) {
  const ConSanMoiReportLayoutOverride expected =
      make_layout_override(override_layout.engine, layout);
  return override_layout.access_record_capacity == expected.access_record_capacity &&
         override_layout.record_replay_logical_access_range_count ==
             expected.record_replay_logical_access_range_count &&
         override_layout.record_replay_dispatch_token_capacity ==
             expected.record_replay_dispatch_token_capacity &&
         override_layout.record_replay_access_dispatch_bank_count ==
             expected.record_replay_access_dispatch_bank_count &&
         override_layout.record_replay_access_owner_bank_count ==
             expected.record_replay_access_owner_bank_count &&
         override_layout.record_replay_address_group_headroom ==
             expected.record_replay_address_group_headroom &&
         override_layout.barrier_record_capacity == expected.barrier_record_capacity &&
         override_layout.atomic_record_capacity == expected.atomic_record_capacity &&
         override_layout.fence_record_capacity == expected.fence_record_capacity &&
         override_layout.diagnostic_capacity == expected.diagnostic_capacity &&
         override_layout.exact_shadow_entry_capacity == expected.exact_shadow_entry_capacity &&
         override_layout.inline_exact_dispatch_bank_count ==
             expected.inline_exact_dispatch_bank_count &&
         override_layout.inline_atomic_release_capacity ==
             expected.inline_atomic_release_capacity &&
         override_layout.inline_acquired_epoch_token_capacity ==
             expected.inline_acquired_epoch_token_capacity &&
         override_layout.inline_causal_snapshot_capacity ==
             expected.inline_causal_snapshot_capacity &&
         override_layout.inline_compact_token_mapping_capacity ==
             expected.inline_compact_token_mapping_capacity &&
         override_layout.sampled_watchpoint_capacity == expected.sampled_watchpoint_capacity &&
         override_layout.sampled_causal_window_capacity ==
             expected.sampled_causal_window_capacity &&
         override_layout.sampled_sync_metadata_capacity ==
             expected.sampled_sync_metadata_capacity &&
         override_layout.sampled_pending_acquire_capacity ==
             expected.sampled_pending_acquire_capacity &&
         override_layout.record_replay_dispatch_tokens_offset ==
             expected.record_replay_dispatch_tokens_offset &&
         override_layout.access_records_offset == expected.access_records_offset &&
         override_layout.barrier_records_offset == expected.barrier_records_offset &&
         override_layout.atomic_records_offset == expected.atomic_records_offset &&
         override_layout.fence_records_offset == expected.fence_records_offset &&
         override_layout.diagnostic_records_offset == expected.diagnostic_records_offset &&
         override_layout.exact_shadow_entries_offset == expected.exact_shadow_entries_offset &&
         override_layout.inline_atomic_release_slots_offset ==
             expected.inline_atomic_release_slots_offset &&
         override_layout.inline_acquired_epoch_token_slots_offset ==
             expected.inline_acquired_epoch_token_slots_offset &&
         override_layout.inline_causal_snapshots_offset ==
             expected.inline_causal_snapshots_offset &&
         override_layout.inline_compact_token_mappings_offset ==
             expected.inline_compact_token_mappings_offset &&
         override_layout.sampled_watchpoints_offset == expected.sampled_watchpoints_offset &&
         override_layout.sampled_causal_windows_offset == expected.sampled_causal_windows_offset &&
         override_layout.sampled_sync_metadata_offset == expected.sampled_sync_metadata_offset &&
         override_layout.sampled_pending_acquires_offset ==
             expected.sampled_pending_acquires_offset &&
         override_layout.required_bytes == expected.required_bytes;
}

[[nodiscard]] bool finalize_plan(ConSanMoiAutoReportPlan &plan, uint64_t cursor) {
  const auto required = util::checked_align_up(cursor, uint64_t{alignof(uint64_t)});
  if (!required || *required > std::numeric_limits<size_t>::max()) {
    plan.reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  plan.required_bytes = *required;
  plan.layout.required_bytes = static_cast<size_t>(*required);
  if (*required > plan.ceiling_bytes) {
    plan.outcome = ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity;
    plan.reason = ConSanMoiAutoReportPlanReason::PerBufferCeiling;
    return true;
  }
  plan.outcome = ConSanMoiAutoReportPlanOutcome::Complete;
  plan.reason = ConSanMoiAutoReportPlanReason::None;
  plan.layout.valid = true;
  return true;
}

[[nodiscard]] bool plan_record_replay(const ConSanMoiAutoReportInventory &inventory,
                                      ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  if (!checked_capacity(inventory.access_range_count,
                        layout.record_replay_logical_access_range_count)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  if (inventory.access_range_count == 0u) {
    layout.record_replay_access_dispatch_bank_count = 1u;
    layout.record_replay_access_owner_bank_count = 1u;
    layout.record_replay_address_group_headroom = 1u;
  } else {
    if (inventory.record_replay_dispatch_token_capacity == 0u ||
        inventory.record_replay_dispatch_token_capacity >
            kConSanMoiRecordReplayMaximumDispatchTokenCount ||
        (inventory.record_replay_dispatch_token_capacity &
         (inventory.record_replay_dispatch_token_capacity - 1u)) != 0u ||
        inventory.record_replay_access_dispatch_bank_count == 0u ||
        inventory.record_replay_access_dispatch_bank_count >
            kConSanMoiRecordReplayMaximumDispatchBankCount ||
        (inventory.record_replay_access_dispatch_bank_count &
         (inventory.record_replay_access_dispatch_bank_count - 1u)) != 0u ||
        inventory.record_replay_access_owner_bank_count == 0u ||
        inventory.record_replay_access_owner_bank_count >
            kConSanMoiRecordReplayMaximumOwnerBankCount ||
        (inventory.record_replay_access_owner_bank_count &
         (inventory.record_replay_access_owner_bank_count - 1u)) != 0u ||
        inventory.record_replay_address_group_headroom == 0u ||
        inventory.record_replay_address_group_headroom >
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
        (inventory.record_replay_address_group_headroom &
         (inventory.record_replay_address_group_headroom - 1u)) != 0u) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
    if (!checked_capacity(inventory.record_replay_dispatch_token_capacity,
                          layout.record_replay_dispatch_token_capacity) ||
        !checked_capacity(inventory.record_replay_access_dispatch_bank_count,
                          layout.record_replay_access_dispatch_bank_count) ||
        !checked_capacity(inventory.record_replay_access_owner_bank_count,
                          layout.record_replay_access_owner_bank_count) ||
        !checked_capacity(inventory.record_replay_address_group_headroom,
                          layout.record_replay_address_group_headroom)) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
  }
  uint64_t access_record_count = 0;
  const auto dispatch_records =
      util::checked_mul(inventory.access_range_count,
                        static_cast<uint64_t>(layout.record_replay_access_dispatch_bank_count));
  const auto owner_records =
      dispatch_records
          ? util::checked_mul(*dispatch_records,
                              static_cast<uint64_t>(layout.record_replay_access_owner_bank_count))
          : std::nullopt;
  const auto address_group_records =
      owner_records
          ? util::checked_mul(*owner_records, inventory.record_replay_address_group_headroom)
          : std::nullopt;
  const auto hash_records =
      address_group_records ? util::checked_mul(*address_group_records,
                                                uint64_t{kConSanMoiRecordReplayHashTableHeadroom})
                            : std::nullopt;
  if (!hash_records) {
    plan.reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  access_record_count = *hash_records;
  if (access_record_count != 0u) {
    if (access_record_count > uint64_t{1} << 31u) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
    access_record_count = std::bit_ceil(access_record_count);
  }
  if (!checked_capacity(access_record_count, layout.access_record_capacity) ||
      !checked_capacity(inventory.barrier_event_count, layout.barrier_record_capacity) ||
      !checked_capacity(inventory.atomic_event_count, layout.atomic_record_capacity) ||
      !checked_capacity(inventory.fence_event_count, layout.fence_record_capacity) ||
      !checked_capacity(inventory.diagnostic_count, layout.diagnostic_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_region(layout.record_replay_dispatch_token_capacity, sizeof(uint64_t),
                       alignof(uint64_t), cursor, layout.record_replay_dispatch_tokens_offset) &&
         append_region(access_record_count, sizeof(ConSanMoiAccessRecord),
                       alignof(ConSanMoiAccessRecord), cursor, layout.access_records_offset) &&
         append_region(inventory.barrier_event_count, sizeof(ConSanMoiBarrierRecord),
                       alignof(ConSanMoiBarrierRecord), cursor, layout.barrier_records_offset) &&
         append_region(inventory.atomic_event_count, sizeof(ConSanMoiAtomicRecord),
                       alignof(ConSanMoiAtomicRecord), cursor, layout.atomic_records_offset) &&
         append_region(inventory.fence_event_count, sizeof(ConSanMoiFenceRecord),
                       alignof(ConSanMoiFenceRecord), cursor, layout.fence_records_offset) &&
         append_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                       alignof(ConSanMoiDiagnosticRecord), cursor,
                       layout.diagnostic_records_offset);
}

[[nodiscard]] bool plan_sampled(const ConSanMoiAutoReportInventory &inventory,
                                ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  if (!checked_capacity(inventory.diagnostic_count, layout.diagnostic_capacity) ||
      !checked_capacity(inventory.sampled_range_bank_count,
                        layout.sampled_causal_window_capacity) ||
      !checked_capacity(inventory.sampled_watchpoint_count, layout.sampled_watchpoint_capacity) ||
      !checked_capacity(inventory.sampled_range_bank_count,
                        layout.sampled_sync_metadata_capacity) ||
      !checked_capacity(inventory.sampled_range_bank_count,
                        layout.sampled_pending_acquire_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                       alignof(ConSanMoiDiagnosticRecord), cursor,
                       layout.diagnostic_records_offset) &&
         append_region(inventory.sampled_range_bank_count, sizeof(ConSanMoiSampledCausalWindow),
                       alignof(ConSanMoiSampledCausalWindow), cursor,
                       layout.sampled_causal_windows_offset) &&
         append_region(inventory.sampled_watchpoint_count, sizeof(uint64_t), alignof(uint64_t),
                       cursor, layout.sampled_watchpoints_offset) &&
         append_region(inventory.sampled_range_bank_count,
                       sizeof(ConSanMoiSampledSyncMetadataPacked),
                       alignof(ConSanMoiSampledSyncMetadataPacked), cursor,
                       layout.sampled_sync_metadata_offset) &&
         append_region(inventory.sampled_range_bank_count,
                       sizeof(ConSanMoiSampledPendingAcquireSlot),
                       alignof(ConSanMoiSampledPendingAcquireSlot), cursor,
                       layout.sampled_pending_acquires_offset);
}

[[nodiscard]] bool plan_inline(const ConSanMoiAutoReportInventory &inventory,
                               ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  const auto rounded_lds_bytes = util::checked_add(
      inventory.inline_lds_bytes, uint64_t{consan_moi_exact_shadow::granule_bytes - 1u});
  if (!rounded_lds_bytes) {
    plan.reason = ConSanMoiAutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  const uint64_t exact_shadow_cells = *rounded_lds_bytes / consan_moi_exact_shadow::granule_bytes;
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
  if (!checked_capacity(inventory.diagnostic_count, layout.diagnostic_capacity) ||
      !checked_capacity(*exact_shadow_count, layout.exact_shadow_entry_capacity) ||
      !checked_capacity(inventory.inline_atomic_release_count,
                        layout.inline_atomic_release_capacity) ||
      !checked_capacity(inventory.inline_causal_snapshot_count,
                        layout.inline_causal_snapshot_capacity) ||
      !checked_capacity(inventory.inline_compact_token_mapping_count,
                        layout.inline_compact_token_mapping_capacity) ||
      !checked_capacity(inventory.inline_acquired_epoch_token_count,
                        layout.inline_acquired_epoch_token_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                       alignof(ConSanMoiDiagnosticRecord), cursor,
                       layout.diagnostic_records_offset) &&
         append_region(*exact_shadow_count, sizeof(ConSanMoiInlineExactShadowSlot),
                       alignof(ConSanMoiInlineExactShadowSlot), cursor,
                       layout.exact_shadow_entries_offset) &&
         append_region(inventory.inline_atomic_release_count,
                       sizeof(ConSanMoiInlineAtomicReleaseSlot),
                       alignof(ConSanMoiInlineAtomicReleaseSlot), cursor,
                       layout.inline_atomic_release_slots_offset) &&
         append_region(inventory.inline_causal_snapshot_count,
                       sizeof(ConSanMoiInlineCausalSnapshot),
                       alignof(ConSanMoiInlineCausalSnapshot), cursor,
                       layout.inline_causal_snapshots_offset) &&
         append_region(inventory.inline_compact_token_mapping_count,
                       sizeof(ConSanMoiCompactDiagnosticTokenMapping),
                       alignof(ConSanMoiCompactDiagnosticTokenMapping), cursor,
                       layout.inline_compact_token_mappings_offset) &&
         append_region(inventory.inline_acquired_epoch_token_count,
                       sizeof(ConSanMoiInlineAcquiredEpochTokenSlot),
                       alignof(ConSanMoiInlineAcquiredEpochTokenSlot), cursor,
                       layout.inline_acquired_epoch_token_slots_offset);
}

} // namespace

ConSanMoiAutoReportPlan plan_consan_moi_auto_report(const ConSanMoiAutoReportInventory &inventory) {
  ConSanMoiAutoReportPlan plan;
  plan.engine = inventory.engine;
  plan.ceiling_bytes = consan_moi_auto_report_buffer_ceiling_bytes(inventory.engine);
  alias_unused_regions(plan.layout, sizeof(ConSanMoiReportHeader));
  uint64_t cursor = sizeof(ConSanMoiReportHeader);

  bool planned = false;
  switch (inventory.engine) {
  case ConSanMoiEngine::RecordReplay:
    planned = plan_record_replay(inventory, plan, cursor);
    break;
  case ConSanMoiEngine::Sampled:
    planned = plan_sampled(inventory, plan, cursor);
    break;
  case ConSanMoiEngine::InlineShadow:
    planned = plan_inline(inventory, plan, cursor);
    break;
  }
  if (!planned)
    return plan;
  if (!finalize_plan(plan, cursor))
    return plan;

  const size_t end = plan.layout.required_bytes;
  switch (inventory.engine) {
  case ConSanMoiEngine::RecordReplay:
    plan.layout.exact_shadow_entries_offset = end;
    plan.layout.inline_atomic_release_slots_offset = end;
    plan.layout.inline_causal_snapshots_offset = end;
    plan.layout.inline_compact_token_mappings_offset = end;
    plan.layout.inline_acquired_epoch_token_slots_offset = end;
    plan.layout.sampled_causal_windows_offset = end;
    plan.layout.sampled_watchpoints_offset = end;
    plan.layout.sampled_sync_metadata_offset = end;
    plan.layout.sampled_pending_acquires_offset = end;
    break;
  case ConSanMoiEngine::Sampled:
    plan.layout.record_replay_dispatch_tokens_offset = end;
    plan.layout.access_records_offset = end;
    plan.layout.barrier_records_offset = end;
    plan.layout.atomic_records_offset = end;
    plan.layout.fence_records_offset = end;
    plan.layout.exact_shadow_entries_offset = end;
    plan.layout.inline_atomic_release_slots_offset = end;
    plan.layout.inline_causal_snapshots_offset = end;
    plan.layout.inline_compact_token_mappings_offset = end;
    plan.layout.inline_acquired_epoch_token_slots_offset = end;
    break;
  case ConSanMoiEngine::InlineShadow:
    plan.layout.record_replay_dispatch_tokens_offset = end;
    plan.layout.access_records_offset = end;
    plan.layout.barrier_records_offset = end;
    plan.layout.atomic_records_offset = end;
    plan.layout.fence_records_offset = end;
    plan.layout.sampled_causal_windows_offset = end;
    plan.layout.sampled_watchpoints_offset = end;
    plan.layout.sampled_sync_metadata_offset = end;
    plan.layout.sampled_pending_acquires_offset = end;
    break;
  }
  return plan;
}

ConSanMoiAutoReportInventory
fit_consan_moi_record_replay_auto_report_inventory(ConSanMoiAutoReportInventory inventory) {
  if (inventory.engine != ConSanMoiEngine::RecordReplay)
    return inventory;

  const uint64_t static_barriers = inventory.barrier_event_count;
  const uint64_t static_atomics = inventory.atomic_event_count;
  const uint64_t static_fences = inventory.fence_event_count;
  const uint64_t static_diagnostics = inventory.diagnostic_count;
  const auto expanded_count = [](uint64_t count, uint64_t headroom) {
    return util::saturating_mul(count, headroom);
  };
  const auto expanded_candidate = [&](uint64_t headroom) {
    ConSanMoiAutoReportInventory candidate = inventory;
    candidate.barrier_event_count = expanded_count(static_barriers, headroom);
    candidate.atomic_event_count = expanded_count(static_atomics, headroom);
    candidate.fence_event_count = expanded_count(static_fences, headroom);
    if (candidate.record_replay_bank_count_adaptive) {
      uint64_t diagnostic_count = util::saturating_mul(
          candidate.access_range_count, candidate.record_replay_access_dispatch_bank_count);
      diagnostic_count =
          util::saturating_mul(diagnostic_count, candidate.record_replay_access_owner_bank_count);
      diagnostic_count =
          util::saturating_mul(diagnostic_count, candidate.record_replay_address_group_headroom);
      candidate.diagnostic_count = std::max(static_diagnostics, diagnostic_count);
    }
    return candidate;
  };

  for (;;) {
    uint64_t headroom = kConSanMoiRecordReplayDynamicEventHeadroom;
    for (;;) {
      ConSanMoiAutoReportInventory candidate = expanded_candidate(headroom);
      const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(candidate);
      if (plan.complete())
        return candidate;
      if (plan.outcome != ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity ||
          plan.reason != ConSanMoiAutoReportPlanReason::PerBufferCeiling) {
        return candidate;
      }
      if (headroom == 1u)
        break;
      headroom = std::max<uint64_t>(headroom / 2u, 1u);
    }

    if (!inventory.record_replay_bank_count_adaptive) {
      return expanded_candidate(1u);
    }
    if (inventory.record_replay_access_dispatch_bank_count != 1u ||
        inventory.record_replay_access_owner_bank_count != 1u) {
      inventory.record_replay_access_dispatch_bank_count =
          std::max<uint64_t>(inventory.record_replay_access_dispatch_bank_count / 2u, 1u);
      inventory.record_replay_access_owner_bank_count =
          std::max<uint64_t>(inventory.record_replay_access_owner_bank_count / 2u, 1u);
      continue;
    }
    if (inventory.record_replay_address_group_headroom != 1u) {
      inventory.record_replay_address_group_headroom =
          std::max<uint64_t>(inventory.record_replay_address_group_headroom / 2u, 1u);
      continue;
    }
    return expanded_candidate(1u);
  }
}

ConSanMoiAutoReportInventory
fit_consan_moi_sampled_auto_report_inventory(ConSanMoiAutoReportInventory inventory) {
  if (inventory.engine != ConSanMoiEngine::Sampled || !inventory.sampled_bank_count_adaptive ||
      inventory.access_range_count == 0u ||
      inventory.sampled_range_bank_count != inventory.sampled_watchpoint_count ||
      inventory.sampled_range_bank_count % inventory.access_range_count != 0u) {
    return inventory;
  }

  uint64_t bank_count = inventory.sampled_range_bank_count / inventory.access_range_count;
  if (bank_count == 0u || bank_count > 8u || (bank_count & (bank_count - 1u)) != 0u)
    return inventory;

  while (bank_count > 1u) {
    const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
    if (plan.complete() ||
        plan.outcome != ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity ||
        plan.reason != ConSanMoiAutoReportPlanReason::PerBufferCeiling) {
      return inventory;
    }
    bank_count /= 2u;
    inventory.sampled_range_bank_count = inventory.access_range_count * bank_count;
    inventory.sampled_watchpoint_count = inventory.sampled_range_bank_count;
  }
  return inventory;
}

std::optional<ConSanMoiReportLayoutOverride>
consan_moi_auto_report_layout_override(const ConSanMoiAutoReportPlan &plan) {
  if (!plan.complete() || !plan.layout.valid || plan.required_bytes != plan.layout.required_bytes)
    return std::nullopt;
  return make_layout_override(plan.engine, plan.layout);
}

ConSanMoiReportBufferLayout
consan_moi_report_layout_from_override(const ConSanMoiReportLayoutOverride &override_layout,
                                       ConSanMoiEngine engine, uint64_t report_buffer_size) {
  if (override_layout.engine != engine || override_layout.required_bytes > report_buffer_size)
    return {};

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = engine;
  switch (engine) {
  case ConSanMoiEngine::RecordReplay: {
    const bool empty_access_layout =
        override_layout.access_record_capacity == 0u &&
        override_layout.record_replay_logical_access_range_count == 0u &&
        override_layout.record_replay_dispatch_token_capacity == 0u &&
        override_layout.record_replay_access_dispatch_bank_count == 1u &&
        override_layout.record_replay_access_owner_bank_count == 1u &&
        override_layout.record_replay_address_group_headroom == 1u;
    if ((!empty_access_layout &&
         (override_layout.record_replay_dispatch_token_capacity == 0u ||
          override_layout.record_replay_dispatch_token_capacity >
              kConSanMoiRecordReplayMaximumDispatchTokenCount ||
          (override_layout.record_replay_dispatch_token_capacity &
           (override_layout.record_replay_dispatch_token_capacity - 1u)) != 0u)) ||
        override_layout.record_replay_access_dispatch_bank_count == 0u ||
        override_layout.record_replay_access_dispatch_bank_count >
            kConSanMoiRecordReplayMaximumDispatchBankCount ||
        (override_layout.record_replay_access_dispatch_bank_count &
         (override_layout.record_replay_access_dispatch_bank_count - 1u)) != 0u ||
        override_layout.record_replay_access_owner_bank_count == 0u ||
        override_layout.record_replay_access_owner_bank_count >
            kConSanMoiRecordReplayMaximumOwnerBankCount ||
        (override_layout.record_replay_access_owner_bank_count &
         (override_layout.record_replay_access_owner_bank_count - 1u)) != 0u ||
        override_layout.record_replay_address_group_headroom == 0u ||
        override_layout.record_replay_address_group_headroom >
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave ||
        (override_layout.record_replay_address_group_headroom &
         (override_layout.record_replay_address_group_headroom - 1u)) != 0u) {
      return {};
    }
    const uint64_t bank_count =
        static_cast<uint64_t>(override_layout.record_replay_access_dispatch_bank_count) *
        override_layout.record_replay_access_owner_bank_count;
    const auto minimum_access_capacity = util::checked_mul(
        static_cast<uint64_t>(override_layout.record_replay_logical_access_range_count),
        bank_count);
    const auto grouped_access_capacity =
        minimum_access_capacity
            ? util::checked_mul(
                  *minimum_access_capacity,
                  static_cast<uint64_t>(override_layout.record_replay_address_group_headroom))
            : std::nullopt;
    if (!grouped_access_capacity ||
        *grouped_access_capacity > override_layout.access_record_capacity)
      return {};
    inventory.access_range_count = override_layout.record_replay_logical_access_range_count;
    inventory.record_replay_dispatch_token_capacity =
        override_layout.record_replay_dispatch_token_capacity;
    inventory.record_replay_access_dispatch_bank_count =
        override_layout.record_replay_access_dispatch_bank_count;
    inventory.record_replay_access_owner_bank_count =
        override_layout.record_replay_access_owner_bank_count;
    inventory.record_replay_address_group_headroom =
        override_layout.record_replay_address_group_headroom;
    inventory.barrier_event_count = override_layout.barrier_record_capacity;
    inventory.atomic_event_count = override_layout.atomic_record_capacity;
    inventory.fence_event_count = override_layout.fence_record_capacity;
    inventory.diagnostic_count = override_layout.diagnostic_capacity;
    break;
  }
  case ConSanMoiEngine::Sampled:
    inventory.diagnostic_count = override_layout.diagnostic_capacity;
    inventory.sampled_range_bank_count = override_layout.sampled_causal_window_capacity;
    inventory.sampled_watchpoint_count = override_layout.sampled_watchpoint_capacity;
    break;
  case ConSanMoiEngine::InlineShadow:
    if (override_layout.inline_exact_dispatch_bank_count == 0u ||
        override_layout.inline_exact_dispatch_bank_count >
            kConSanMoiInlineMaximumDispatchBankCount ||
        (override_layout.inline_exact_dispatch_bank_count &
         (override_layout.inline_exact_dispatch_bank_count - 1u)) != 0u ||
        override_layout.exact_shadow_entry_capacity %
            override_layout.inline_exact_dispatch_bank_count) {
      return {};
    }
    inventory.diagnostic_count = override_layout.diagnostic_capacity;
    inventory.inline_lds_bytes =
        (static_cast<uint64_t>(override_layout.exact_shadow_entry_capacity) /
         override_layout.inline_exact_dispatch_bank_count) *
        consan_moi_exact_shadow::granule_bytes;
    inventory.inline_atomic_release_count = override_layout.inline_atomic_release_capacity;
    inventory.inline_causal_snapshot_count = override_layout.inline_causal_snapshot_capacity;
    inventory.inline_compact_token_mapping_count =
        override_layout.inline_compact_token_mapping_capacity;
    inventory.inline_acquired_epoch_token_count =
        override_layout.inline_acquired_epoch_token_capacity;
    break;
  }
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  if (!plan.complete() || !layout_override_matches(override_layout, plan.layout))
    return {};
  return plan.layout;
}

} // namespace rocjitsu
