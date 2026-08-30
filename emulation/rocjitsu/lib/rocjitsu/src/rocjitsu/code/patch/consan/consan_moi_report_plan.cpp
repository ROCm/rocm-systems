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

void add_saturating(uint64_t &count, uint64_t increment) {
  count = increment > std::numeric_limits<uint64_t>::max() - count
              ? std::numeric_limits<uint64_t>::max()
              : count + increment;
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

ConSanEvidenceRequirementReason
validate_moi_evidence_intents(const ConSanEvidenceIntentPlan &plan,
                              ConSanCapabilityEngine expected_engine) {
  if (plan.reason != ConSanEvidenceRequirementReason::None)
    return plan.reason;
  if (!plan.well_formed())
    return ConSanEvidenceRequirementReason::InvalidIntentPayload;
  if (plan.engine != expected_engine)
    return ConSanEvidenceRequirementReason::WrongEngine;
  return ConSanEvidenceRequirementReason::None;
}

std::vector<const ConSanEvidenceIntent *>
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

void publish_moi_evidence_requirements(ConSanMoiEvidenceRequirements &requirements,
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

} // namespace

bool ConSanMoiEvidenceRequirements::common_well_formed(ConSanMoiEngine expected_engine) const {
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

namespace {

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

bool consan_evidence_requirements_well_formed(const ConSanEvidenceRequirements &requirements) {
  return std::visit([](const auto &value) { return value.well_formed(); }, requirements);
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
