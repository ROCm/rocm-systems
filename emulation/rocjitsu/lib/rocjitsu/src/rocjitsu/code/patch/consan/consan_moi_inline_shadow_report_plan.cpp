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
  return plan_moi_report_regions(
      {moi_report_region<ConSanMoiDiagnosticRecord>(inventory.diagnostic_count,
                                                    layout.diagnostic_capacity,
                                                    layout.diagnostic_records_offset),
       moi_report_region<ConSanMoiInlineExactShadowSlot>(*exact_shadow_count,
                                                         layout.exact_shadow_entry_capacity,
                                                         layout.exact_shadow_entries_offset),
       moi_report_region<ConSanMoiInlineAtomicReleaseSlot>(
           inventory.inline_atomic_release_count, layout.inline_atomic_release_capacity,
           layout.inline_atomic_release_slots_offset),
       moi_report_region<ConSanMoiInlineCausalSnapshot>(inventory.inline_causal_snapshot_count,
                                                        layout.inline_causal_snapshot_capacity,
                                                        layout.inline_causal_snapshots_offset),
       moi_report_region<ConSanMoiCompactDiagnosticTokenMapping>(
           inventory.inline_compact_token_mapping_count,
           layout.inline_compact_token_mapping_capacity,
           layout.inline_compact_token_mappings_offset),
       moi_report_region<ConSanMoiInlineAcquiredEpochTokenSlot>(
           inventory.inline_acquired_epoch_token_count, layout.inline_acquired_epoch_token_capacity,
           layout.inline_acquired_epoch_token_slots_offset)},
      plan, cursor);
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
namespace {

[[nodiscard]] const ConSanAccessInventorySite *
find_inventory_access_range(const ProgramInventory &inventory, const SemanticSiteId &range_id) {
  const auto site = std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
    return std::ranges::find(candidate.ranges, range_id, &ConSanAccessRange::id) !=
           candidate.ranges.end();
  });
  return site == inventory.access_sites().end() ? nullptr : &*site;
}

} // namespace

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

bool ConSanInlineShadowEvidenceRequirements::well_formed() const {
  const uint64_t expected_ordering_capacity = std::max<uint64_t>(
      sizing_inventory.atomic_event_count, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  const uint64_t expected_acquired_epoch_capacity = std::max<uint64_t>(
      expected_ordering_capacity, kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity);
  return required_lds_aperture_bytes == sizing_inventory.inline_lds_bytes &&
         sizing_inventory.inline_atomic_release_count == expected_ordering_capacity &&
         sizing_inventory.inline_causal_snapshot_count == expected_ordering_capacity &&
         sizing_inventory.inline_acquired_epoch_token_count == expected_acquired_epoch_capacity &&
         sizing_inventory.inline_compact_token_mapping_count <=
             sizing_inventory.access_range_count &&
         sizing_inventory.inline_diagnostic_count_adaptive &&
         common_well_formed(ConSanMoiEngine::InlineShadow);
}

ConSanInlineShadowEvidenceRequirements
plan_consan_inline_shadow_evidence(const ProgramInventory &program_inventory,
                                   const ConSanEvidenceIntentPlan &evidence_intents,
                                   const ConSanInlineShadowCapacityPolicy &capacity_policy) {
  ConSanInlineShadowEvidenceRequirements requirements;
  requirements.reason = consan_moi_impl::validate_moi_evidence_intents(
      evidence_intents, ConSanCapabilityEngine::InlineShadow);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::InlineShadow;
  const std::vector<const ConSanEvidenceIntent *> retained_accesses =
      consan_moi_impl::accumulate_moi_evidence_counts(
          evidence_intents, capacity_policy.maximum_access_probe_count, inventory);
  bool requires_full_lds_aperture = false;
  uint64_t declared_lds_extent = 0;
  uint64_t native_static_extent = 0;
  for (const ConSanEvidenceIntent *intent : retained_accesses) {
    inventory.inline_compact_token_mapping_count =
        util::saturating_add(inventory.inline_compact_token_mapping_count, uint64_t{1});
    for (const SemanticSiteId &range_id : intent->semantic_sites) {
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
        const ConSanKernelInfo *owner = program_inventory.find_kernel_by_descriptor(owner_offset);
        if (owner == nullptr || !owner->declared_group_segment_bytes) {
          requirements.reason = ConSanEvidenceRequirementReason::MissingInventoryFact;
          return requirements;
        }
        declared_lds_extent =
            std::max<uint64_t>(declared_lds_extent, *owner->declared_group_segment_bytes);
        requires_full_lds_aperture |= owner->has_dynamic_lds;
      }
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

  requirements.required_lds_aperture_bytes = inventory.inline_lds_bytes;
  consan_moi_impl::publish_moi_evidence_requirements(requirements, std::move(inventory),
                                                     capacity_policy.caller_ceiling_bytes);
  return requirements;
}

} // namespace rocjitsu
