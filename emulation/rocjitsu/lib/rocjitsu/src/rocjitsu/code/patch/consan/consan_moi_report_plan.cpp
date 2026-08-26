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
      plan.reason = ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow;
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
  const uint64_t sampled_sync_slot_count =
      std::max(inventory.sampled_range_bank_count, inventory.sampled_sync_slot_count);
  // Access-only objects never create deferred acquires, so their historical
  // one-to-one pending table is sufficient.  Once an atomic is admitted, each
  // causal slot must be able to retain every wave's acquire until the
  // associated later access executes.
  const uint64_t sampled_pending_acquire_owner_bank_count =
      inventory.atomic_event_count == 0u ? 1u : kConSanMoiSampledPendingAcquireOwnerBankCount;
  const auto sampled_pending_acquire_count =
      util::checked_mul(sampled_sync_slot_count, sampled_pending_acquire_owner_bank_count);
  if (!sampled_pending_acquire_count) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  if (!checked_capacity(inventory.diagnostic_count, layout.diagnostic_capacity) ||
      !checked_capacity(sampled_sync_slot_count, layout.sampled_causal_window_capacity) ||
      !checked_capacity(inventory.sampled_watchpoint_count, layout.sampled_watchpoint_capacity) ||
      !checked_capacity(sampled_sync_slot_count, layout.sampled_sync_metadata_capacity) ||
      !checked_capacity(*sampled_pending_acquire_count, layout.sampled_pending_acquire_capacity)) {
    plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return append_region(inventory.diagnostic_count, sizeof(ConSanMoiDiagnosticRecord),
                       alignof(ConSanMoiDiagnosticRecord), cursor,
                       layout.diagnostic_records_offset) &&
         append_region(sampled_sync_slot_count, sizeof(ConSanMoiSampledCausalWindow),
                       alignof(ConSanMoiSampledCausalWindow), cursor,
                       layout.sampled_causal_windows_offset) &&
         append_region(inventory.sampled_watchpoint_count, sizeof(uint64_t), alignof(uint64_t),
                       cursor, layout.sampled_watchpoints_offset) &&
         append_region(sampled_sync_slot_count, sizeof(ConSanMoiSampledSyncMetadataPacked),
                       alignof(ConSanMoiSampledSyncMetadataPacked), cursor,
                       layout.sampled_sync_metadata_offset) &&
         append_region(*sampled_pending_acquire_count, sizeof(ConSanMoiSampledPendingAcquireSlot),
                       alignof(ConSanMoiSampledPendingAcquireSlot), cursor,
                       layout.sampled_pending_acquires_offset);
}

[[nodiscard]] bool plan_inline(const ConSanMoiAutoReportInventory &inventory,
                               ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  auto &layout = plan.layout;
  // Provision one external slot per LDS byte. Objects containing subword
  // traffic use those slots directly; objects that retain four-byte external
  // cells use a conservative subset of the same bounded allocation.
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

bool ConSanRecordReplayEvidenceRequirements::well_formed() const {
  if (reason != ConSanEvidenceRequirementReason::None ||
      schema != ConSanEvidenceSchema::RecordReplay ||
      boundedness != ConSanEvidenceBoundedness::BoundedFirstLight ||
      loss_severity != ConSanEvidenceLossSeverity::InvalidatesCompleteness ||
      delivery_scope != ConSanRuntimeResourceScope::Executable ||
      sizing_inventory.engine != ConSanMoiEngine::RecordReplay ||
      abi_plan.engine != ConSanMoiEngine::RecordReplay ||
      !runtime_requirements.host_device_visible_memory ||
      !runtime_requirements.host_device_coherent_memory ||
      !runtime_requirements.device_atomic_publication || !runtime_requirements.executable_binding ||
      runtime_requirements.max_workgroup_lds_bytes ||
      runtime_requirements.dispatch_segment_binding ||
      !runtime_requirements.minimum_report_allocation_bytes ||
      *runtime_requirements.minimum_report_allocation_bytes != abi_plan.required_bytes) {
    return false;
  }
  if (abi_plan.outcome == ConSanMoiAutoReportPlanOutcome::Count ||
      abi_plan.reason == ConSanMoiAutoReportPlanReason::Count)
    return false;
  if (plan_consan_moi_auto_report(sizing_inventory, abi_plan.ceiling_bytes) != abi_plan)
    return false;
  if (abi_plan.complete())
    return abi_plan.reason == ConSanMoiAutoReportPlanReason::None && abi_plan.layout.valid &&
           abi_plan.layout.required_bytes == abi_plan.required_bytes;
  return abi_plan.reason != ConSanMoiAutoReportPlanReason::None && !abi_plan.layout.valid;
}

namespace {

[[nodiscard]] bool common_moi_evidence_is_well_formed(
    ConSanEvidenceRequirementReason reason, ConSanMoiEngine expected_engine,
    const RuntimeCapabilityRequirements &runtime_requirements,
    const ConSanMoiAutoReportInventory &sizing_inventory, const ConSanMoiAutoReportPlan &abi_plan) {
  if (reason != ConSanEvidenceRequirementReason::None ||
      sizing_inventory.engine != expected_engine || abi_plan.engine != expected_engine ||
      !runtime_requirements.host_device_visible_memory ||
      !runtime_requirements.host_device_coherent_memory ||
      !runtime_requirements.device_atomic_publication || !runtime_requirements.executable_binding ||
      runtime_requirements.max_workgroup_lds_bytes ||
      runtime_requirements.dispatch_segment_binding ||
      !runtime_requirements.minimum_report_allocation_bytes ||
      *runtime_requirements.minimum_report_allocation_bytes != abi_plan.required_bytes ||
      abi_plan.outcome == ConSanMoiAutoReportPlanOutcome::Count ||
      abi_plan.reason == ConSanMoiAutoReportPlanReason::Count) {
    return false;
  }
  if (plan_consan_moi_auto_report(sizing_inventory, abi_plan.ceiling_bytes) != abi_plan)
    return false;
  if (abi_plan.complete())
    return abi_plan.reason == ConSanMoiAutoReportPlanReason::None && abi_plan.layout.valid &&
           abi_plan.layout.required_bytes == abi_plan.required_bytes;
  return abi_plan.reason != ConSanMoiAutoReportPlanReason::None && !abi_plan.layout.valid;
}

void add_saturating(uint64_t &count, uint64_t increment) {
  count = increment > std::numeric_limits<uint64_t>::max() - count
              ? std::numeric_limits<uint64_t>::max()
              : count + increment;
}

[[nodiscard]] bool intent_covers_only(const ConSanProbeIntent &intent,
                                      ConSanSemanticSiteDomain domain) {
  return std::ranges::all_of(intent.covered_semantic_sites,
                             [&](const SemanticSiteId &site) { return site.domain == domain; });
}

[[nodiscard]] const ConSanAccessInventorySite *
find_inventory_access_range(const ProgramInventory &inventory, const SemanticSiteId &range_id) {
  const auto site = std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
    return std::ranges::find(candidate.ranges, range_id, &ConSanAccessRange::id) !=
           candidate.ranges.end();
  });
  return site == inventory.access_sites().end() ? nullptr : &*site;
}

} // namespace

bool ConSanSampledEvidenceRequirements::well_formed() const {
  if (schema != ConSanEvidenceSchema::Sampled ||
      boundedness != ConSanEvidenceBoundedness::BoundedSampled ||
      loss_severity != ConSanEvidenceLossSeverity::InvalidatesCompleteness ||
      delivery_scope != ConSanRuntimeResourceScope::Executable ||
      sizing_inventory.diagnostic_count != sizing_inventory.access_range_count ||
      sizing_inventory.sampled_bank_count_adaptive != (sizing_inventory.access_range_count != 0u)) {
    return false;
  }
  if (sizing_inventory.access_range_count == 0u) {
    if (sizing_inventory.sampled_range_bank_count != 0u ||
        sizing_inventory.sampled_sync_slot_count != 0u ||
        sizing_inventory.sampled_watchpoint_count != 0u) {
      return false;
    }
  } else {
    if (sizing_inventory.sampled_range_bank_count % sizing_inventory.access_range_count != 0u)
      return false;
    const uint64_t banks_per_range =
        sizing_inventory.sampled_range_bank_count / sizing_inventory.access_range_count;
    if (banks_per_range == 0u || banks_per_range > 8u ||
        (banks_per_range & (banks_per_range - 1u)) != 0u) {
      return false;
    }
    const uint64_t expected_slots = util::saturating_add(sizing_inventory.sampled_range_bank_count,
                                                         sizing_inventory.atomic_event_count);
    if (sizing_inventory.sampled_sync_slot_count != expected_slots ||
        sizing_inventory.sampled_watchpoint_count != expected_slots) {
      return false;
    }
  }
  return common_moi_evidence_is_well_formed(reason, ConSanMoiEngine::Sampled, runtime_requirements,
                                            sizing_inventory, abi_plan);
}

bool ConSanInlineShadowEvidenceRequirements::well_formed() const {
  const uint64_t expected_ordering_capacity = std::max<uint64_t>(
      sizing_inventory.atomic_event_count, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  const uint64_t expected_acquired_epoch_capacity = std::max<uint64_t>(
      expected_ordering_capacity, kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity);
  return schema == ConSanEvidenceSchema::InlineShadow &&
         boundedness == ConSanEvidenceBoundedness::ExactOnDevice &&
         loss_severity == ConSanEvidenceLossSeverity::InvalidatesCompleteness &&
         delivery_scope == ConSanRuntimeResourceScope::Executable &&
         required_lds_aperture_bytes == sizing_inventory.inline_lds_bytes &&
         sizing_inventory.inline_atomic_release_count == expected_ordering_capacity &&
         sizing_inventory.inline_causal_snapshot_count == expected_ordering_capacity &&
         sizing_inventory.inline_acquired_epoch_token_count == expected_acquired_epoch_capacity &&
         sizing_inventory.inline_compact_token_mapping_count <=
             sizing_inventory.access_range_count &&
         sizing_inventory.inline_diagnostic_count_adaptive &&
         common_moi_evidence_is_well_formed(reason, ConSanMoiEngine::InlineShadow,
                                            runtime_requirements, sizing_inventory, abi_plan);
}

bool ConSanSuperColliderEvidenceRequirements::well_formed() const {
  return reason == ConSanEvidenceRequirementReason::None &&
         schema == ConSanEvidenceSchema::SuperCollider &&
         boundedness == ConSanEvidenceBoundedness::StickyMarker &&
         loss_severity == ConSanEvidenceLossSeverity::InvalidatesCompleteness &&
         delivery_scope == ConSanRuntimeResourceScope::Executable &&
         (marker_bytes == 0u || marker_bytes == sizeof(uint32_t)) &&
         runtime_requirements.host_device_visible_memory &&
         runtime_requirements.host_device_coherent_memory &&
         !runtime_requirements.device_atomic_publication &&
         runtime_requirements.minimum_report_allocation_bytes == marker_bytes &&
         !runtime_requirements.max_workgroup_lds_bytes && runtime_requirements.executable_binding &&
         !runtime_requirements.dispatch_segment_binding;
}

ConSanEvidenceSchema
consan_evidence_requirements_schema(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &value) { return value.schema; }, requirements);
}

bool consan_evidence_requirements_well_formed(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &value) { return value.well_formed(); }, requirements);
}

ConSanRecordReplayEvidenceRequirements
plan_consan_record_replay_evidence(const ConSanObservationPlan &observation_plan,
                                   const ConSanRecordReplayCapacityPolicy &capacity_policy) {
  ConSanRecordReplayEvidenceRequirements requirements;
  if (!observation_plan.valid()) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }
  if (observation_plan.engine != ConSanCapabilityEngine::RecordReplay) {
    requirements.reason = ConSanEvidenceRequirementReason::WrongEngine;
    return requirements;
  }

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  uint64_t selected_access_probe_count = 0;
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    switch (intent.kind) {
    case ConSanProbeIntentKind::AccessRecord:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::Access)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      if (capacity_policy.maximum_access_probe_count &&
          selected_access_probe_count >= *capacity_policy.maximum_access_probe_count) {
        break;
      }
      ++selected_access_probe_count;
      add_saturating(inventory.access_range_count, intent.covered_semantic_sites.size());
      break;
    case ConSanProbeIntentKind::BarrierRecord:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.barrier_event_count, 1u);
      break;
    case ConSanProbeIntentKind::AtomicRecord:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.atomic_event_count, 1u);
      break;
    case ConSanProbeIntentKind::FenceRecord:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.fence_event_count, 1u);
      break;
    case ConSanProbeIntentKind::AtomicAddressCapture:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      break;
    case ConSanProbeIntentKind::RedundantAccessObservation:
    case ConSanProbeIntentKind::SampledAccess:
    case ConSanProbeIntentKind::ExactShadowAccess:
    case ConSanProbeIntentKind::SampledBarrierEpoch:
    case ConSanProbeIntentKind::ExactBarrierEpoch:
    case ConSanProbeIntentKind::SampledAtomicOrdering:
    case ConSanProbeIntentKind::ExactAtomicOrdering:
    case ConSanProbeIntentKind::Count:
      requirements.reason = ConSanEvidenceRequirementReason::UnexpectedIntentKind;
      return requirements;
    }
  }
  const bool has_evidence = inventory.access_range_count != 0u ||
                            inventory.barrier_event_count != 0u ||
                            inventory.atomic_event_count != 0u || inventory.fence_event_count != 0u;
  inventory.diagnostic_count =
      has_evidence ? std::max<uint64_t>(inventory.access_range_count, 1u) : 0u;
  inventory.record_replay_bank_count_adaptive = inventory.access_range_count != 0u;
  inventory = fit_consan_moi_record_replay_auto_report_inventory(
      inventory, capacity_policy.caller_ceiling_bytes);

  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.device_atomic_publication = true;
  requirements.runtime_requirements.executable_binding = true;
  requirements.sizing_inventory = inventory;
  requirements.abi_plan =
      plan_consan_moi_auto_report(inventory, capacity_policy.caller_ceiling_bytes);
  requirements.runtime_requirements.minimum_report_allocation_bytes =
      requirements.abi_plan.required_bytes;
  requirements.reason = ConSanEvidenceRequirementReason::None;
  return requirements;
}

ConSanSampledEvidenceRequirements
plan_consan_sampled_evidence(const ConSanObservationPlan &observation_plan,
                             const ConSanSampledCapacityPolicy &capacity_policy) {
  ConSanSampledEvidenceRequirements requirements;
  if (!observation_plan.valid()) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }
  if (observation_plan.engine != ConSanCapabilityEngine::Sampled) {
    requirements.reason = ConSanEvidenceRequirementReason::WrongEngine;
    return requirements;
  }

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::Sampled;
  uint64_t selected_access_probe_count = 0;
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    switch (intent.kind) {
    case ConSanProbeIntentKind::SampledAccess:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::Access)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      if (capacity_policy.maximum_access_probe_count &&
          selected_access_probe_count >= *capacity_policy.maximum_access_probe_count) {
        break;
      }
      ++selected_access_probe_count;
      add_saturating(inventory.access_range_count, intent.covered_semantic_sites.size());
      break;
    case ConSanProbeIntentKind::SampledBarrierEpoch:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.barrier_event_count, intent.covered_semantic_sites.size());
      break;
    case ConSanProbeIntentKind::AtomicAddressCapture:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      break;
    case ConSanProbeIntentKind::SampledAtomicOrdering:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.atomic_event_count, 1u);
      break;
    case ConSanProbeIntentKind::RedundantAccessObservation:
    case ConSanProbeIntentKind::AccessRecord:
    case ConSanProbeIntentKind::ExactShadowAccess:
    case ConSanProbeIntentKind::BarrierRecord:
    case ConSanProbeIntentKind::ExactBarrierEpoch:
    case ConSanProbeIntentKind::AtomicRecord:
    case ConSanProbeIntentKind::ExactAtomicOrdering:
    case ConSanProbeIntentKind::FenceRecord:
    case ConSanProbeIntentKind::Count:
      requirements.reason = ConSanEvidenceRequirementReason::UnexpectedIntentKind;
      return requirements;
    }
  }

  constexpr uint64_t kSampledBanksPerLogicalRange = 8u;
  const uint64_t access_banks =
      util::saturating_mul(inventory.access_range_count, kSampledBanksPerLogicalRange);
  inventory.sampled_range_bank_count = access_banks;
  const uint64_t sampled_slots =
      access_banks == 0u ? 0u : util::saturating_add(access_banks, inventory.atomic_event_count);
  inventory.sampled_sync_slot_count = sampled_slots;
  inventory.sampled_watchpoint_count = sampled_slots;
  inventory.sampled_bank_count_adaptive = inventory.access_range_count != 0u;
  inventory.diagnostic_count = inventory.access_range_count == 0u
                                   ? 0u
                                   : std::max<uint64_t>(inventory.access_range_count, 1u);
  inventory =
      fit_consan_moi_sampled_auto_report_inventory(inventory, capacity_policy.caller_ceiling_bytes);

  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.device_atomic_publication = true;
  requirements.runtime_requirements.executable_binding = true;
  requirements.sizing_inventory = inventory;
  requirements.abi_plan =
      plan_consan_moi_auto_report(inventory, capacity_policy.caller_ceiling_bytes);
  requirements.runtime_requirements.minimum_report_allocation_bytes =
      requirements.abi_plan.required_bytes;
  requirements.reason = ConSanEvidenceRequirementReason::None;
  return requirements;
}

ConSanInlineShadowEvidenceRequirements
plan_consan_inline_shadow_evidence(const ProgramInventory &program_inventory,
                                   const ConSanObservationPlan &observation_plan,
                                   const ConSanInlineShadowCapacityPolicy &capacity_policy) {
  ConSanInlineShadowEvidenceRequirements requirements;
  if (!observation_plan.valid()) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }
  if (observation_plan.engine != ConSanCapabilityEngine::InlineShadow) {
    requirements.reason = ConSanEvidenceRequirementReason::WrongEngine;
    return requirements;
  }

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::InlineShadow;
  uint64_t selected_access_probe_count = 0;
  bool requires_full_lds_aperture = false;
  uint64_t declared_lds_extent = 0;
  uint64_t native_static_extent = 0;
  const auto kernels = program_inventory.kernels();
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    switch (intent.kind) {
    case ConSanProbeIntentKind::ExactShadowAccess: {
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::Access)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      if (capacity_policy.maximum_access_probe_count &&
          selected_access_probe_count >= *capacity_policy.maximum_access_probe_count) {
        break;
      }
      ++selected_access_probe_count;
      add_saturating(inventory.access_range_count, intent.covered_semantic_sites.size());
      add_saturating(inventory.inline_compact_token_mapping_count, 1u);
      for (const SemanticSiteId &range_id : intent.covered_semantic_sites) {
        const ConSanAccessInventorySite *site =
            find_inventory_access_range(program_inventory, range_id);
        if (!site) {
          requirements.reason = ConSanEvidenceRequirementReason::MissingInventoryFact;
          return requirements;
        }
        requires_full_lds_aperture |= site->origin == ConSanAccessOrigin::Flat;
        if (const auto range = std::ranges::find(site->ranges, range_id, &ConSanAccessRange::id);
            range != site->ranges.end() && range->static_byte_offset &&
            *range->static_byte_offset >= 0) {
          native_static_extent =
              std::max(native_static_extent,
                       util::saturating_add(static_cast<uint64_t>(*range->static_byte_offset),
                                            static_cast<uint64_t>(range->byte_width)));
        }

        // Group-FLAT addressing already selects the complete architectural
        // aperture. It does not need a kernel-owner join merely to recover a
        // smaller fixed descriptor declaration, and shared helper ownership
        // may intentionally remain unresolved at this stage.
        if (site->origin == ConSanAccessOrigin::Flat)
          continue;

        std::vector<uint64_t> owners = site->execution_owner_descriptor_file_offsets;
        if (owners.empty() && site->container.kernel_descriptor_file_offset)
          owners.push_back(*site->container.kernel_descriptor_file_offset);
        if (owners.empty()) {
          requirements.reason = ConSanEvidenceRequirementReason::MissingInventoryFact;
          return requirements;
        }
        for (uint64_t owner_offset : owners) {
          const auto owner =
              std::ranges::find(kernels, owner_offset, &ConSanKernelInfo::descriptor_file_offset);
          if (owner == kernels.end() || !owner->declared_group_segment_bytes) {
            requirements.reason = ConSanEvidenceRequirementReason::MissingInventoryFact;
            return requirements;
          }
          declared_lds_extent =
              std::max<uint64_t>(declared_lds_extent, *owner->declared_group_segment_bytes);
          requires_full_lds_aperture |= owner->has_dynamic_lds;
        }
      }
      break;
    }
    case ConSanProbeIntentKind::ExactBarrierEpoch:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.barrier_event_count, intent.covered_semantic_sites.size());
      break;
    case ConSanProbeIntentKind::AtomicAddressCapture:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      break;
    case ConSanProbeIntentKind::ExactAtomicOrdering:
      if (!intent_covers_only(intent, ConSanSemanticSiteDomain::SynchronizationEvent)) {
        requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
        return requirements;
      }
      add_saturating(inventory.atomic_event_count, 1u);
      break;
    case ConSanProbeIntentKind::RedundantAccessObservation:
    case ConSanProbeIntentKind::AccessRecord:
    case ConSanProbeIntentKind::SampledAccess:
    case ConSanProbeIntentKind::BarrierRecord:
    case ConSanProbeIntentKind::SampledBarrierEpoch:
    case ConSanProbeIntentKind::AtomicRecord:
    case ConSanProbeIntentKind::SampledAtomicOrdering:
    case ConSanProbeIntentKind::FenceRecord:
    case ConSanProbeIntentKind::Count:
      requirements.reason = ConSanEvidenceRequirementReason::UnexpectedIntentKind;
      return requirements;
    }
  }

  requires_full_lds_aperture |=
      inventory.access_range_count != 0u && declared_lds_extent < native_static_extent;
  const uint64_t full_lds_aperture = capacity_policy.maximum_workgroup_lds_bytes.value_or(
      consan_moi_max_workgroup_lds_bytes(program_inventory.arch()));
  inventory.inline_lds_bytes =
      std::max(declared_lds_extent, requires_full_lds_aperture ? full_lds_aperture : 0u);
  const uint64_t ordering_capacity = std::max<uint64_t>(
      inventory.atomic_event_count, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  inventory.inline_atomic_release_count = ordering_capacity;
  inventory.inline_causal_snapshot_count = ordering_capacity;
  inventory.inline_acquired_epoch_token_count =
      std::max<uint64_t>(ordering_capacity, kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity);
  const uint64_t inline_dispatch_banks =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes);
  const uint64_t diagnostic_headroom = util::saturating_mul(
      util::saturating_mul(inventory.access_range_count, inline_dispatch_banks),
      static_cast<uint64_t>(kConSanMoiInlineShadowDiagnosticHeadroomPerAccess));
  inventory.diagnostic_count =
      std::max({inventory.access_range_count, diagnostic_headroom,
                static_cast<uint64_t>(kConSanMoiInlineShadowDefaultDiagnosticCapacity)});
  inventory.inline_diagnostic_count_adaptive = true;
  inventory =
      fit_consan_moi_inline_auto_report_inventory(inventory, capacity_policy.caller_ceiling_bytes);

  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.device_atomic_publication = true;
  requirements.runtime_requirements.executable_binding = true;
  requirements.required_lds_aperture_bytes = inventory.inline_lds_bytes;
  requirements.sizing_inventory = inventory;
  requirements.abi_plan =
      plan_consan_moi_auto_report(inventory, capacity_policy.caller_ceiling_bytes);
  requirements.runtime_requirements.minimum_report_allocation_bytes =
      requirements.abi_plan.required_bytes;
  requirements.reason = ConSanEvidenceRequirementReason::None;
  return requirements;
}

ConSanSuperColliderEvidenceRequirements
plan_consan_supercollider_evidence(const ConSanObservationPlan &observation_plan) {
  ConSanSuperColliderEvidenceRequirements requirements;
  if (!observation_plan.valid()) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }
  if (observation_plan.engine != ConSanCapabilityEngine::SuperCollider) {
    requirements.reason = ConSanEvidenceRequirementReason::WrongEngine;
    return requirements;
  }

  bool has_observation = false;
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::RedundantAccessObservation) {
      requirements.reason = ConSanEvidenceRequirementReason::UnexpectedIntentKind;
      return requirements;
    }
    if (!intent_covers_only(intent, ConSanSemanticSiteDomain::Access)) {
      requirements.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
      return requirements;
    }
    has_observation = true;
  }
  requirements.marker_bytes = has_observation ? sizeof(uint32_t) : 0u;
  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.minimum_report_allocation_bytes = requirements.marker_bytes;
  requirements.runtime_requirements.executable_binding = true;
  requirements.reason = ConSanEvidenceRequirementReason::None;
  return requirements;
}

ConSanMoiAutoReportPlan plan_consan_moi_auto_report(const ConSanMoiAutoReportInventory &inventory,
                                                    uint64_t caller_ceiling_bytes) {
  ConSanMoiAutoReportPlan plan;
  plan.engine = inventory.engine;
  const uint64_t engine_ceiling = consan_moi_auto_report_buffer_ceiling_bytes(inventory.engine);
  plan.ceiling_bytes =
      caller_ceiling_bytes == 0u ? engine_ceiling : std::min(caller_ceiling_bytes, engine_ceiling);
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
fit_consan_moi_record_replay_auto_report_inventory(ConSanMoiAutoReportInventory inventory,
                                                   uint64_t caller_ceiling_bytes) {
  if (inventory.engine != ConSanMoiEngine::RecordReplay)
    return inventory;

  const uint64_t static_barriers = inventory.barrier_event_count;
  const uint64_t static_atomics = inventory.atomic_event_count;
  const uint64_t static_fences = inventory.fence_event_count;
  const uint64_t static_diagnostics = inventory.diagnostic_count;
  const auto expanded_count = [](uint64_t count, uint64_t headroom) {
    return util::saturating_mul(count, headroom);
  };
  const auto expanded_candidate = [&](uint64_t lane_headroom) {
    ConSanMoiAutoReportInventory candidate = inventory;
    const uint64_t barrier_headroom = std::max<uint64_t>(lane_headroom / 64u, 1u);
    candidate.barrier_event_count = expanded_count(static_barriers, barrier_headroom);
    candidate.atomic_event_count = expanded_count(static_atomics, lane_headroom);
    candidate.fence_event_count = expanded_count(static_fences, lane_headroom);
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
    uint64_t headroom = kConSanMoiRecordReplayDynamicLaneEventHeadroom;
    for (;;) {
      ConSanMoiAutoReportInventory candidate = expanded_candidate(headroom);
      const ConSanMoiAutoReportPlan plan =
          plan_consan_moi_auto_report(candidate, caller_ceiling_bytes);
      if (plan.complete())
        return candidate;
      const bool capacity_limited =
          plan.outcome == ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity &&
          plan.reason == ConSanMoiAutoReportPlanReason::PerBufferCeiling;
      const bool adaptive_abi_capacity_limited =
          plan.outcome == ConSanMoiAutoReportPlanOutcome::Overflow &&
          plan.reason == ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow &&
          inventory.record_replay_bank_count_adaptive;
      if (!capacity_limited && !adaptive_abi_capacity_limited) {
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
    inventory.sampled_sync_slot_count = override_layout.sampled_causal_window_capacity;
    inventory.sampled_watchpoint_count = override_layout.sampled_watchpoint_capacity;
    if (override_layout.sampled_causal_window_capacity != 0u &&
        override_layout.sampled_pending_acquire_capacity /
                override_layout.sampled_causal_window_capacity ==
            kConSanMoiSampledPendingAcquireOwnerBankCount &&
        override_layout.sampled_pending_acquire_capacity %
                override_layout.sampled_causal_window_capacity ==
            0u) {
      // The override carries capacities rather than static event inventory.
      // Reconstruct the atomic-present layout class so exact auto layouts
      // round-trip through the ordinary planner.
      inventory.atomic_event_count = 1u;
    }
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
         override_layout.inline_exact_dispatch_bank_count);
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
