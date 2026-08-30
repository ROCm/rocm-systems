// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"
#include "util/bit.h"

#include <array>
#include <bit>
#include <limits>
#include <utility>

namespace rocjitsu::consan_moi_impl {

bool checked_moi_report_capacity(uint64_t count, uint32_t &capacity) {
  if (count > std::numeric_limits<uint32_t>::max())
    return false;
  capacity = static_cast<uint32_t>(count);
  return true;
}

namespace {

[[nodiscard]] bool append_moi_report_region(uint64_t count, uint64_t element_size,
                                            uint64_t alignment, uint64_t &cursor, size_t &offset) {
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

} // namespace

bool plan_moi_report_regions(std::initializer_list<MoiReportRegionPlan> regions,
                             ConSanMoiAutoReportPlan &plan, uint64_t &cursor) {
  for (const MoiReportRegionPlan &region : regions) {
    if (!checked_moi_report_capacity(region.count, *region.capacity)) {
      plan.reason = ConSanMoiAutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
    if (!append_moi_report_region(region.count, region.element_size, region.alignment, cursor,
                                  *region.offset))
      return false;
  }
  return true;
}

} // namespace rocjitsu::consan_moi_impl

namespace rocjitsu {
namespace {

[[nodiscard]] std::array<size_t *, 15> report_region_offsets(ConSanMoiReportBufferLayout &layout) {
  return {&layout.record_replay_dispatch_tokens_offset,
          &layout.access_records_offset,
          &layout.barrier_records_offset,
          &layout.atomic_records_offset,
          &layout.fence_records_offset,
          &layout.diagnostic_records_offset,
          &layout.exact_shadow_entries_offset,
          &layout.inline_atomic_release_slots_offset,
          &layout.inline_acquired_epoch_token_slots_offset,
          &layout.inline_causal_snapshots_offset,
          &layout.inline_compact_token_mappings_offset,
          &layout.sampled_watchpoints_offset,
          &layout.sampled_causal_windows_offset,
          &layout.sampled_sync_metadata_offset,
          &layout.sampled_pending_acquires_offset};
}

constexpr size_t kUnplannedReportRegionOffset = std::numeric_limits<size_t>::max();

void mark_report_regions_unplanned(ConSanMoiReportBufferLayout &layout) {
  for (size_t *offset : report_region_offsets(layout))
    *offset = kUnplannedReportRegionOffset;
}

void alias_unplanned_report_regions(ConSanMoiReportBufferLayout &layout, size_t alias) {
  for (size_t *offset : report_region_offsets(layout)) {
    if (*offset == kUnplannedReportRegionOffset)
      *offset = alias;
  }
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

[[nodiscard]] bool common_moi_evidence_is_well_formed(
    ConSanEvidenceRequirementReason reason, ConSanMoiEngine expected_engine,
    const RuntimeCapabilityRequirements &runtime_requirements,
    const ConSanMoiAutoReportInventory &sizing_inventory, const ConSanMoiAutoReportPlan &abi_plan) {
  if (reason != ConSanEvidenceRequirementReason::None ||
      sizing_inventory.engine != expected_engine || abi_plan.layout.engine != expected_engine ||
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

template <typename Requirements>
void publish_moi_evidence_requirements(Requirements &requirements,
                                       ConSanMoiAutoReportInventory inventory,
                                       uint64_t caller_ceiling_bytes) {
  requirements.sizing_inventory = std::move(inventory);
  requirements.abi_plan =
      plan_consan_moi_auto_report(requirements.sizing_inventory, caller_ceiling_bytes);
  requirements.runtime_requirements = {
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .minimum_report_allocation_bytes = requirements.abi_plan.required_bytes,
      .executable_binding = true,
  };
  requirements.reason = ConSanEvidenceRequirementReason::None;
}

void add_saturating(uint64_t &count, uint64_t increment) {
  count = increment > std::numeric_limits<uint64_t>::max() - count
              ? std::numeric_limits<uint64_t>::max()
              : count + increment;
}

[[nodiscard]] constexpr bool valid_evidence_engine(ConSanCapabilityEngine engine) {
  return static_cast<uint8_t>(engine) < static_cast<uint8_t>(ConSanCapabilityEngine::Count);
}

[[nodiscard]] constexpr bool valid_evidence_intent_kind(ConSanEvidenceIntentKind kind) {
  return static_cast<uint8_t>(kind) < static_cast<uint8_t>(ConSanEvidenceIntentKind::Count);
}

[[nodiscard]] constexpr bool engine_accepts_evidence_kind(ConSanCapabilityEngine engine,
                                                          ConSanEvidenceIntentKind kind) {
  switch (engine) {
  case ConSanCapabilityEngine::SuperCollider:
    return kind == ConSanEvidenceIntentKind::StickyMarker;
  case ConSanCapabilityEngine::RecordReplay:
    return kind == ConSanEvidenceIntentKind::Access || kind == ConSanEvidenceIntentKind::Barrier ||
           kind == ConSanEvidenceIntentKind::Atomic || kind == ConSanEvidenceIntentKind::Fence ||
           kind == ConSanEvidenceIntentKind::AddressCapture;
  case ConSanCapabilityEngine::Sampled:
  case ConSanCapabilityEngine::InlineShadow:
    return kind == ConSanEvidenceIntentKind::Access || kind == ConSanEvidenceIntentKind::Barrier ||
           kind == ConSanEvidenceIntentKind::Atomic ||
           kind == ConSanEvidenceIntentKind::AddressCapture;
  case ConSanCapabilityEngine::Count:
    break;
  }
  return false;
}

[[nodiscard]] std::optional<ConSanEvidenceIntentKind>
classify_evidence_intent(ConSanCapabilityEngine engine, ConSanProbeIntentKind kind) {
  switch (kind) {
  case ConSanProbeIntentKind::RedundantAccessObservation:
    if (engine == ConSanCapabilityEngine::SuperCollider)
      return ConSanEvidenceIntentKind::StickyMarker;
    break;
  case ConSanProbeIntentKind::AccessRecord:
    if (engine == ConSanCapabilityEngine::RecordReplay)
      return ConSanEvidenceIntentKind::Access;
    break;
  case ConSanProbeIntentKind::SampledAccess:
    if (engine == ConSanCapabilityEngine::Sampled)
      return ConSanEvidenceIntentKind::Access;
    break;
  case ConSanProbeIntentKind::ExactShadowAccess:
    if (engine == ConSanCapabilityEngine::InlineShadow)
      return ConSanEvidenceIntentKind::Access;
    break;
  case ConSanProbeIntentKind::BarrierRecord:
    if (engine == ConSanCapabilityEngine::RecordReplay)
      return ConSanEvidenceIntentKind::Barrier;
    break;
  case ConSanProbeIntentKind::SampledBarrierEpoch:
    if (engine == ConSanCapabilityEngine::Sampled)
      return ConSanEvidenceIntentKind::Barrier;
    break;
  case ConSanProbeIntentKind::ExactBarrierEpoch:
    if (engine == ConSanCapabilityEngine::InlineShadow)
      return ConSanEvidenceIntentKind::Barrier;
    break;
  case ConSanProbeIntentKind::AtomicAddressCapture:
    if (engine == ConSanCapabilityEngine::RecordReplay ||
        engine == ConSanCapabilityEngine::Sampled ||
        engine == ConSanCapabilityEngine::InlineShadow) {
      return ConSanEvidenceIntentKind::AddressCapture;
    }
    break;
  case ConSanProbeIntentKind::AtomicRecord:
    if (engine == ConSanCapabilityEngine::RecordReplay)
      return ConSanEvidenceIntentKind::Atomic;
    break;
  case ConSanProbeIntentKind::SampledAtomicOrdering:
    if (engine == ConSanCapabilityEngine::Sampled)
      return ConSanEvidenceIntentKind::Atomic;
    break;
  case ConSanProbeIntentKind::ExactAtomicOrdering:
    if (engine == ConSanCapabilityEngine::InlineShadow)
      return ConSanEvidenceIntentKind::Atomic;
    break;
  case ConSanProbeIntentKind::FenceRecord:
    if (engine == ConSanCapabilityEngine::RecordReplay)
      return ConSanEvidenceIntentKind::Fence;
    break;
  case ConSanProbeIntentKind::Count:
    break;
  }
  return std::nullopt;
}

[[nodiscard]] constexpr ConSanSemanticSiteDomain
evidence_intent_domain(ConSanEvidenceIntentKind kind) {
  return kind == ConSanEvidenceIntentKind::Access || kind == ConSanEvidenceIntentKind::StickyMarker
             ? ConSanSemanticSiteDomain::Access
             : ConSanSemanticSiteDomain::SynchronizationEvent;
}

[[nodiscard]] uint64_t
expected_evidence_element_count(ConSanCapabilityEngine engine, ConSanEvidenceIntentKind kind,
                                std::span<const SemanticSiteId> semantic_sites) {
  switch (kind) {
  case ConSanEvidenceIntentKind::Access:
    return semantic_sites.size();
  case ConSanEvidenceIntentKind::Barrier:
    return engine == ConSanCapabilityEngine::RecordReplay ? 1u : semantic_sites.size();
  case ConSanEvidenceIntentKind::Atomic:
  case ConSanEvidenceIntentKind::Fence:
  case ConSanEvidenceIntentKind::StickyMarker:
    return 1u;
  case ConSanEvidenceIntentKind::AddressCapture:
  case ConSanEvidenceIntentKind::Count:
    return 0u;
  }
  return 0u;
}

[[nodiscard]] std::vector<const ConSanEvidenceIntent *>
accumulate_moi_evidence_counts(const ConSanEvidenceIntentPlan &plan,
                               std::optional<uint64_t> maximum_access_probe_count,
                               ConSanMoiAutoReportInventory &inventory) {
  std::vector<const ConSanEvidenceIntent *> retained_accesses;
  uint64_t selected_access_probe_count = 0;
  for (const ConSanEvidenceIntent &intent : plan.intents) {
    switch (intent.kind) {
    case ConSanEvidenceIntentKind::Access:
      if (maximum_access_probe_count &&
          selected_access_probe_count >= *maximum_access_probe_count) {
        break;
      }
      ++selected_access_probe_count;
      retained_accesses.push_back(&intent);
      add_saturating(inventory.access_range_count, intent.element_count);
      break;
    case ConSanEvidenceIntentKind::Barrier:
      add_saturating(inventory.barrier_event_count, intent.element_count);
      break;
    case ConSanEvidenceIntentKind::Atomic:
      add_saturating(inventory.atomic_event_count, intent.element_count);
      break;
    case ConSanEvidenceIntentKind::Fence:
      add_saturating(inventory.fence_event_count, intent.element_count);
      break;
    case ConSanEvidenceIntentKind::AddressCapture:
    case ConSanEvidenceIntentKind::StickyMarker:
    case ConSanEvidenceIntentKind::Count:
      break;
    }
  }
  return retained_accesses;
}

[[nodiscard]] const ConSanAccessInventorySite *
find_inventory_access_range(const ProgramInventory &inventory, const SemanticSiteId &range_id) {
  const auto site = std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
    return std::ranges::find(candidate.ranges, range_id, &ConSanAccessRange::id) !=
           candidate.ranges.end();
  });
  return site == inventory.access_sites().end() ? nullptr : &*site;
}

[[nodiscard]] ConSanEvidenceRequirementReason
validate_evidence_intent_input(const ConSanEvidenceIntentPlan &plan,
                               ConSanCapabilityEngine expected_engine) {
  if (plan.reason != ConSanEvidenceRequirementReason::None)
    return plan.reason;
  if (!plan.well_formed())
    return ConSanEvidenceRequirementReason::InvalidIntentPayload;
  if (plan.engine != expected_engine)
    return ConSanEvidenceRequirementReason::WrongEngine;
  return ConSanEvidenceRequirementReason::None;
}

} // namespace

bool ConSanEvidenceIntentPlan::well_formed() const {
  if (reason != ConSanEvidenceRequirementReason::None || !valid_evidence_engine(engine))
    return false;
  for (size_t index = 0; index < intents.size(); ++index) {
    const ConSanEvidenceIntent &intent = intents[index];
    if (intent.source_intent.value != index || !valid_evidence_intent_kind(intent.kind) ||
        !engine_accepts_evidence_kind(engine, intent.kind) || intent.semantic_sites.empty() ||
        std::ranges::any_of(intent.semantic_sites,
                            [](const SemanticSiteId &site) { return !site.valid(); }) ||
        std::ranges::any_of(intent.semantic_sites,
                            [&](const SemanticSiteId &site) {
                              return site.domain != evidence_intent_domain(intent.kind);
                            }) ||
        intent.element_count !=
            expected_evidence_element_count(engine, intent.kind, intent.semantic_sites)) {
      return false;
    }
  }
  return true;
}

ConSanEvidenceIntentPlan
plan_consan_evidence_intents(const ConSanObservationPlan &observation_plan) {
  ConSanEvidenceIntentPlan plan;
  plan.engine = observation_plan.engine;
  if (!observation_plan.valid()) {
    plan.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return plan;
  }

  plan.intents.reserve(observation_plan.probe_intents.size());
  for (const ConSanProbeIntent &probe : observation_plan.probe_intents) {
    const std::optional<ConSanEvidenceIntentKind> kind =
        classify_evidence_intent(observation_plan.engine, probe.kind);
    if (!kind) {
      plan.intents.clear();
      plan.reason = ConSanEvidenceRequirementReason::UnexpectedIntentKind;
      return plan;
    }
    ConSanEvidenceIntent intent{
        .source_intent = probe.id,
        .kind = *kind,
        .semantic_sites = probe.covered_semantic_sites,
    };
    intent.element_count =
        expected_evidence_element_count(plan.engine, intent.kind, intent.semantic_sites);
    if (std::ranges::any_of(intent.semantic_sites, [&](const SemanticSiteId &site) {
          return site.domain != evidence_intent_domain(intent.kind);
        })) {
      plan.intents.clear();
      plan.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
      return plan;
    }
    plan.intents.push_back(std::move(intent));
  }
  plan.reason = ConSanEvidenceRequirementReason::None;
  if (!plan.well_formed()) {
    plan.intents.clear();
    plan.reason = ConSanEvidenceRequirementReason::InvalidIntentPayload;
  }
  return plan;
}

bool ConSanRecordReplayEvidenceRequirements::well_formed() const {
  return common_moi_evidence_is_well_formed(reason, ConSanMoiEngine::RecordReplay,
                                            runtime_requirements, sizing_inventory, abi_plan);
}

bool ConSanSampledEvidenceRequirements::well_formed() const {
  if (sizing_inventory.diagnostic_count != sizing_inventory.access_range_count ||
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
  return required_lds_aperture_bytes == sizing_inventory.inline_lds_bytes &&
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
  if (reason != ConSanEvidenceRequirementReason::None)
    return false;
  if (mode == ConSanSuperColliderEvidenceMode::TrapOnly) {
    return marker_bytes == 0u && runtime_requirements == RuntimeCapabilityRequirements{};
  }
  return mode == ConSanSuperColliderEvidenceMode::StickyMarker &&
         (marker_bytes == 0u || marker_bytes == sizeof(uint32_t)) &&
         runtime_requirements.host_device_visible_memory &&
         runtime_requirements.host_device_coherent_memory &&
         !runtime_requirements.device_atomic_publication &&
         runtime_requirements.minimum_report_allocation_bytes == marker_bytes &&
         !runtime_requirements.max_workgroup_lds_bytes && runtime_requirements.executable_binding &&
         !runtime_requirements.dispatch_segment_binding;
}

bool consan_evidence_requirements_well_formed(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &value) { return value.well_formed(); }, requirements);
}

ConSanRecordReplayEvidenceRequirements
plan_consan_record_replay_evidence(const ConSanEvidenceIntentPlan &evidence_intents,
                                   const ConSanRecordReplayCapacityPolicy &capacity_policy) {
  ConSanRecordReplayEvidenceRequirements requirements;
  requirements.reason =
      validate_evidence_intent_input(evidence_intents, ConSanCapabilityEngine::RecordReplay);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  (void)accumulate_moi_evidence_counts(evidence_intents, capacity_policy.maximum_access_probe_count,
                                       inventory);
  const bool has_evidence = inventory.access_range_count != 0u ||
                            inventory.barrier_event_count != 0u ||
                            inventory.atomic_event_count != 0u || inventory.fence_event_count != 0u;
  inventory.diagnostic_count =
      has_evidence ? std::max<uint64_t>(inventory.access_range_count, 1u) : 0u;
  inventory.record_replay_bank_count_adaptive = inventory.access_range_count != 0u;
  inventory = fit_consan_moi_record_replay_auto_report_inventory(
      inventory, capacity_policy.caller_ceiling_bytes);

  publish_moi_evidence_requirements(requirements, std::move(inventory),
                                    capacity_policy.caller_ceiling_bytes);
  return requirements;
}

ConSanSampledEvidenceRequirements
plan_consan_sampled_evidence(const ConSanEvidenceIntentPlan &evidence_intents,
                             const ConSanSampledCapacityPolicy &capacity_policy) {
  ConSanSampledEvidenceRequirements requirements;
  requirements.reason =
      validate_evidence_intent_input(evidence_intents, ConSanCapabilityEngine::Sampled);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::Sampled;
  (void)accumulate_moi_evidence_counts(evidence_intents, capacity_policy.maximum_access_probe_count,
                                       inventory);

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

  publish_moi_evidence_requirements(requirements, std::move(inventory),
                                    capacity_policy.caller_ceiling_bytes);
  return requirements;
}

ConSanInlineShadowEvidenceRequirements
plan_consan_inline_shadow_evidence(const ProgramInventory &program_inventory,
                                   const ConSanEvidenceIntentPlan &evidence_intents,
                                   const ConSanInlineShadowCapacityPolicy &capacity_policy) {
  ConSanInlineShadowEvidenceRequirements requirements;
  requirements.reason =
      validate_evidence_intent_input(evidence_intents, ConSanCapabilityEngine::InlineShadow);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;

  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::InlineShadow;
  const std::vector<const ConSanEvidenceIntent *> retained_accesses =
      accumulate_moi_evidence_counts(evidence_intents, capacity_policy.maximum_access_probe_count,
                                     inventory);
  bool requires_full_lds_aperture = false;
  uint64_t declared_lds_extent = 0;
  uint64_t native_static_extent = 0;
  for (const ConSanEvidenceIntent *intent : retained_accesses) {
    add_saturating(inventory.inline_compact_token_mapping_count, 1u);
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
  publish_moi_evidence_requirements(requirements, std::move(inventory),
                                    capacity_policy.caller_ceiling_bytes);
  return requirements;
}

ConSanSuperColliderEvidenceRequirements
plan_consan_supercollider_evidence(const ConSanEvidenceIntentPlan &evidence_intents,
                                   ConSanSuperColliderEvidenceMode mode) {
  ConSanSuperColliderEvidenceRequirements requirements;
  requirements.mode = mode;
  requirements.reason =
      validate_evidence_intent_input(evidence_intents, ConSanCapabilityEngine::SuperCollider);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;
  if (mode != ConSanSuperColliderEvidenceMode::TrapOnly &&
      mode != ConSanSuperColliderEvidenceMode::StickyMarker) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }

  if (mode == ConSanSuperColliderEvidenceMode::TrapOnly) {
    requirements.reason = ConSanEvidenceRequirementReason::None;
    return requirements;
  }

  requirements.marker_bytes = evidence_intents.intents.empty() ? 0u : sizeof(uint32_t);
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
  plan.layout.engine = inventory.engine;
  const uint64_t engine_ceiling = consan_moi_auto_report_buffer_ceiling_bytes(inventory.engine);
  plan.ceiling_bytes =
      caller_ceiling_bytes == 0u ? engine_ceiling : std::min(caller_ceiling_bytes, engine_ceiling);
  mark_report_regions_unplanned(plan.layout);
  uint64_t cursor = sizeof(ConSanMoiReportHeader);

  if (!consan_moi_impl::moi_mode_operations(inventory.engine)
           .plan_report_layout(inventory, plan, cursor)) {
    alias_unplanned_report_regions(plan.layout, sizeof(ConSanMoiReportHeader));
    return plan;
  }
  if (!finalize_plan(plan, cursor)) {
    alias_unplanned_report_regions(plan.layout, sizeof(ConSanMoiReportHeader));
    return plan;
  }

  alias_unplanned_report_regions(plan.layout, plan.layout.required_bytes);
  return plan;
}

ConSanMoiReportBufferLayout
revalidate_consan_moi_report_layout(const ConSanMoiReportBufferLayout &candidate,
                                    ConSanMoiEngine engine, uint64_t report_buffer_size) {
  if (candidate.engine != engine || !candidate.valid ||
      candidate.required_bytes > report_buffer_size)
    return {};

  const auto inventory =
      consan_moi_impl::moi_mode_operations(engine).reconstruct_report_inventory(candidate);
  if (!inventory)
    return {};
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(*inventory);
  if (!plan.complete() || candidate != plan.layout)
    return {};
  return plan.layout;
}

} // namespace rocjitsu
