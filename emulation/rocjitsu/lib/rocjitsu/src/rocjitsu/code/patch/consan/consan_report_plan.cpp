// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_report.h"
#include "rocjitsu/code/patch/consan/consan_report_planning.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <utility>

namespace rocjitsu::consan::detail {

bool checked_report_capacity(uint64_t count, uint32_t &capacity) {
  if (count > std::numeric_limits<uint32_t>::max())
    return false;
  capacity = static_cast<uint32_t>(count);
  return true;
}

namespace {

[[nodiscard]] bool append_report_region(uint64_t count, uint64_t element_size, uint64_t alignment,
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

void add_saturating(uint64_t &count, uint64_t increment) {
  count = increment > std::numeric_limits<uint64_t>::max() - count
              ? std::numeric_limits<uint64_t>::max()
              : count + increment;
}

} // namespace

bool plan_report_regions(std::initializer_list<ReportRegionPlan> regions, AutoReportPlan &plan,
                         uint64_t &cursor) {
  for (const ReportRegionPlan &region : regions) {
    if (!checked_report_capacity(region.count, *region.capacity)) {
      plan.reason = AutoReportPlanReason::AbiCapacityOverflow;
      return false;
    }
    if (!append_report_region(region.count, region.element_size, region.alignment, cursor,
                              *region.offset))
      return false;
  }
  return true;
}

void publish_evidence_requirements(ReportRequirements &requirements, AutoReportInventory inventory,
                                   uint64_t caller_ceiling_bytes) {
  requirements.sizing_inventory = std::move(inventory);
  requirements.abi_plan = plan_auto_report(requirements.sizing_inventory, caller_ceiling_bytes);
  requirements.runtime_requirements = {
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .minimum_report_allocation_bytes = requirements.abi_plan.required_bytes,
      .executable_binding = true,
  };
  requirements.reason = EvidenceRequirementReason::None;
}

} // namespace rocjitsu::consan::detail

namespace rocjitsu::consan {
namespace {

[[nodiscard]] std::array<size_t *, 4> report_region_offsets(ReportBufferLayout &layout) {
  return {&layout.watchpoints_offset, &layout.causal_windows_offset, &layout.sync_metadata_offset,
          &layout.pending_acquires_offset};
}

constexpr size_t kUnplannedReportRegionOffset = std::numeric_limits<size_t>::max();

void mark_report_regions_unplanned(ReportBufferLayout &layout) {
  for (size_t *offset : report_region_offsets(layout))
    *offset = kUnplannedReportRegionOffset;
}

void alias_unplanned_report_regions(ReportBufferLayout &layout, size_t alias) {
  for (size_t *offset : report_region_offsets(layout)) {
    if (*offset == kUnplannedReportRegionOffset)
      *offset = alias;
  }
}

[[nodiscard]] bool finalize_plan(AutoReportPlan &plan, uint64_t cursor) {
  const auto required = util::checked_align_up(cursor, uint64_t{alignof(uint64_t)});
  if (!required || *required > std::numeric_limits<size_t>::max()) {
    plan.reason = AutoReportPlanReason::ByteSizeOverflow;
    return false;
  }
  plan.required_bytes = *required;
  plan.layout.required_bytes = static_cast<size_t>(*required);
  if (*required > plan.ceiling_bytes) {
    plan.outcome = AutoReportPlanOutcome::InsufficientReportCapacity;
    plan.reason = AutoReportPlanReason::PerBufferCeiling;
    return true;
  }
  plan.outcome = AutoReportPlanOutcome::Complete;
  plan.reason = AutoReportPlanReason::None;
  plan.layout.valid = true;
  return true;
}

} // namespace

namespace {

using Probe = ProbeIntentKind;
using Evidence = EvidenceIntentKind;

[[nodiscard]] constexpr std::optional<Evidence>
classify_evidence_intent(const ModeProbeVocabulary &vocabulary, Probe kind) {
  if (kind == vocabulary.access)
    return vocabulary.sticky_access_evidence ? Evidence::StickyMarker : Evidence::Access;
  if (kind == Probe::AtomicAddressCapture && vocabulary.address_capture)
    return Evidence::AddressCapture;
  if (kind == vocabulary.barrier && kind != Probe::Count)
    return Evidence::Barrier;
  if (kind == vocabulary.atomic && kind != Probe::Count)
    return Evidence::Atomic;
  return std::nullopt;
}

[[nodiscard]] constexpr SemanticSiteDomain evidence_intent_domain(EvidenceIntentKind kind) {
  return kind == EvidenceIntentKind::Access || kind == EvidenceIntentKind::StickyMarker
             ? SemanticSiteDomain::Access
             : SemanticSiteDomain::SynchronizationEvent;
}

[[nodiscard]] uint64_t evidence_element_count(const ModeProbeVocabulary &vocabulary, Evidence kind,
                                              std::span<const SemanticSiteId> semantic_sites) {
  switch (kind) {
  case Evidence::Access:
    return semantic_sites.size();
  case Evidence::Barrier:
    return vocabulary.barrier_elements_per_semantic_site ? semantic_sites.size() : 1u;
  case Evidence::Atomic:
  case Evidence::StickyMarker:
    return 1u;
  case Evidence::AddressCapture:
  case Evidence::Count:
    return 0u;
  }
  return 0u;
}

} // namespace

std::optional<EvidenceIntentKind> evidence_intent_kind(Mode mode, ProbeIntentKind kind) {
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(mode);
  return vocabulary == nullptr ? std::nullopt : classify_evidence_intent(*vocabulary, kind);
}

std::optional<uint64_t> evidence_element_count(Mode mode, const ProbeIntent &intent) {
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(mode);
  const auto kind = evidence_intent_kind(mode, intent.kind);
  if (vocabulary == nullptr || !kind)
    return std::nullopt;
  return evidence_element_count(*vocabulary, *kind, intent.covered_semantic_sites);
}

EvidenceRequirementReason detail::validate_evidence_intents(const ObservationPlan &plan,
                                                            Mode expected_mode) {
  if (!plan.valid())
    return EvidenceRequirementReason::InvalidObservationPlan;
  if (plan.mode != expected_mode)
    return EvidenceRequirementReason::WrongMode;
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(plan.mode);
  if (vocabulary == nullptr)
    return EvidenceRequirementReason::InvalidObservationPlan;
  for (const ProbeIntent &probe : plan.probe_intents) {
    const std::optional<Evidence> kind = evidence_intent_kind(plan.mode, probe.kind);
    if (!kind)
      return EvidenceRequirementReason::UnexpectedIntentKind;
    if (std::ranges::any_of(probe.covered_semantic_sites, [&](const SemanticSiteId &site) {
          return site.domain != evidence_intent_domain(*kind);
        }))
      return EvidenceRequirementReason::InvalidIntentPayload;
  }
  return EvidenceRequirementReason::None;
}

std::vector<const ProbeIntent *>
detail::accumulate_evidence_counts(const ObservationPlan &plan,
                                   std::optional<uint64_t> maximum_access_probe_count,
                                   AutoReportInventory &inventory) {
  std::vector<const ProbeIntent *> retained_accesses;
  const ModeProbeVocabulary *vocabulary = mode_probe_vocabulary(plan.mode);
  if (vocabulary == nullptr)
    return retained_accesses;
  uint64_t selected_access_probe_count = 0;
  for (const ProbeIntent &intent : plan.probe_intents) {
    const Evidence kind = evidence_intent_kind(plan.mode, intent.kind).value_or(Evidence::Count);
    const uint64_t element_count = evidence_element_count(plan.mode, intent).value_or(0u);
    switch (kind) {
    case Evidence::Access:
      if (maximum_access_probe_count && selected_access_probe_count >= *maximum_access_probe_count)
        break;
      ++selected_access_probe_count;
      retained_accesses.push_back(&intent);
      add_saturating(inventory.access_range_count, element_count);
      break;
    case Evidence::Barrier:
      add_saturating(inventory.barrier_event_count, element_count);
      break;
    case Evidence::Atomic:
      add_saturating(inventory.atomic_event_count, element_count);
      break;
    case Evidence::AddressCapture:
    case Evidence::StickyMarker:
    case Evidence::Count:
      break;
    }
  }
  return retained_accesses;
}

bool evidence_requirements_well_formed(const EvidenceRequirements &requirements) {
  return std::visit([](const auto &value) { return value.well_formed(); }, requirements);
}

AutoReportPlan plan_auto_report(const AutoReportInventory &inventory,
                                uint64_t caller_ceiling_bytes) {
  AutoReportPlan plan;
  const uint64_t mode_ceiling = kOrdinaryAutoReportBufferCeilingBytes;
  plan.ceiling_bytes =
      caller_ceiling_bytes == 0u ? mode_ceiling : std::min(caller_ceiling_bytes, mode_ceiling);
  mark_report_regions_unplanned(plan.layout);
  uint64_t cursor = sizeof(ReportHeader);

  if (!detail::plan_report_layout(inventory, plan, cursor)) {
    alias_unplanned_report_regions(plan.layout, sizeof(ReportHeader));
    return plan;
  }
  if (!finalize_plan(plan, cursor)) {
    alias_unplanned_report_regions(plan.layout, sizeof(ReportHeader));
    return plan;
  }

  alias_unplanned_report_regions(plan.layout, plan.layout.required_bytes);
  return plan;
}

ReportBufferLayout revalidate_report_layout(const ReportBufferLayout &candidate,
                                            uint64_t report_buffer_size) {
  if (!candidate.valid || candidate.required_bytes > report_buffer_size)
    return {};

  const auto inventory = detail::reconstruct_report_inventory(candidate);
  if (!inventory)
    return {};
  // Reconstruct against the actual allocation. Adaptive geometry selected by
  // the original plan is part of the reconstructed inventory, so validation
  // checks that exact choice rather than adapting it again to this tighter
  // ceiling.
  const AutoReportPlan plan = plan_auto_report(*inventory, report_buffer_size);
  if (!plan.complete() || candidate != plan.layout)
    return {};
  return plan.layout;
}

} // namespace rocjitsu::consan

namespace rocjitsu::consan::detail {

bool plan_report_layout(const AutoReportInventory &inventory, AutoReportPlan &plan,
                        uint64_t &cursor) {
  auto &layout = plan.layout;
  const uint64_t sync_slot_count = std::max(inventory.range_bank_count, inventory.sync_slot_count);
  // Access-only objects never create deferred acquires. Once an atomic is
  // admitted, each causal slot must retain every wave's acquire until the
  // associated later access executes.
  const uint64_t pending_owner_bank_count =
      inventory.atomic_event_count == 0u ? 1u : kPendingAcquireOwnerBankCount;
  const auto pending_acquire_count = util::checked_mul(sync_slot_count, pending_owner_bank_count);
  if (!pending_acquire_count) {
    plan.reason = AutoReportPlanReason::AbiCapacityOverflow;
    return false;
  }
  return plan_report_regions(
      {report_region<CausalWindow>(sync_slot_count, layout.causal_window_capacity,
                                   layout.causal_windows_offset),
       report_region<uint64_t>(inventory.watchpoint_count, layout.watchpoint_capacity,
                               layout.watchpoints_offset),
       report_region<SyncMetadataPacked>(sync_slot_count, layout.sync_metadata_capacity,
                                         layout.sync_metadata_offset),
       report_region<PendingAcquireSlot>(*pending_acquire_count, layout.pending_acquire_capacity,
                                         layout.pending_acquires_offset)},
      plan, cursor);
}

std::optional<AutoReportInventory>
reconstruct_report_inventory(const ReportBufferLayout &candidate) {
  AutoReportInventory inventory;
  inventory.range_bank_count = candidate.causal_window_capacity;
  inventory.sync_slot_count = candidate.causal_window_capacity;
  inventory.watchpoint_count = candidate.watchpoint_capacity;
  if (candidate.causal_window_capacity != 0u &&
      candidate.pending_acquire_capacity / candidate.causal_window_capacity ==
          kPendingAcquireOwnerBankCount &&
      candidate.pending_acquire_capacity % candidate.causal_window_capacity == 0u) {
    // Reconstruct the atomic-present layout class from the retained
    // capacities so exact auto layouts round-trip through the planner.
    inventory.atomic_event_count = 1u;
  }
  return inventory;
}

} // namespace rocjitsu::consan::detail

namespace rocjitsu::consan {

AutoReportInventory fit_auto_report_inventory(AutoReportInventory inventory,
                                              uint64_t caller_ceiling_bytes) {
  if (!inventory.bank_count_adaptive || inventory.access_range_count == 0u ||
      inventory.watchpoint_count < inventory.range_bank_count ||
      inventory.range_bank_count % inventory.access_range_count != 0u) {
    return inventory;
  }

  const uint64_t reserved_sync_slots =
      std::max(inventory.sync_slot_count, inventory.range_bank_count) - inventory.range_bank_count;
  const uint64_t extra_watchpoints = inventory.watchpoint_count - inventory.range_bank_count;
  uint64_t bank_count = inventory.range_bank_count / inventory.access_range_count;
  if (bank_count == 0u || bank_count > 1024u || (bank_count & (bank_count - 1u)) != 0u)
    return inventory;

  while (bank_count > 1u) {
    const AutoReportPlan plan = plan_auto_report(inventory, caller_ceiling_bytes);
    if (plan.complete() || plan.outcome != AutoReportPlanOutcome::InsufficientReportCapacity ||
        plan.reason != AutoReportPlanReason::PerBufferCeiling) {
      return inventory;
    }
    bank_count /= 2u;
    inventory.range_bank_count = inventory.access_range_count * bank_count;
    inventory.sync_slot_count =
        util::saturating_add(inventory.range_bank_count, reserved_sync_slots);
    inventory.watchpoint_count =
        util::saturating_add(inventory.range_bank_count, extra_watchpoints);
  }
  return inventory;
}

bool ReportRequirements::well_formed() const {
  if (sizing_inventory.bank_count_adaptive != (sizing_inventory.access_range_count != 0u)) {
    return false;
  }
  if (sizing_inventory.access_range_count == 0u) {
    if (sizing_inventory.range_bank_count != 0u || sizing_inventory.sync_slot_count != 0u ||
        sizing_inventory.watchpoint_count != 0u) {
      return false;
    }
  } else {
    if (sizing_inventory.range_bank_count % sizing_inventory.access_range_count != 0u)
      return false;
    const uint64_t banks_per_range =
        sizing_inventory.range_bank_count / sizing_inventory.access_range_count;
    if (banks_per_range == 0u || banks_per_range > 1024u || !std::has_single_bit(banks_per_range))
      return false;
    const uint64_t expected_slots = util::saturating_add(sizing_inventory.range_bank_count,
                                                         sizing_inventory.atomic_event_count);
    if (sizing_inventory.sync_slot_count != expected_slots ||
        sizing_inventory.watchpoint_count != expected_slots) {
      return false;
    }
  }
  if (reason != EvidenceRequirementReason::None ||
      !runtime_requirements.host_device_visible_memory ||
      !runtime_requirements.host_device_coherent_memory ||
      !runtime_requirements.device_atomic_publication || !runtime_requirements.executable_binding ||
      runtime_requirements.max_workgroup_lds_bytes ||
      runtime_requirements.dispatch_segment_binding ||
      !runtime_requirements.minimum_report_allocation_bytes ||
      *runtime_requirements.minimum_report_allocation_bytes != abi_plan.required_bytes ||
      abi_plan.outcome == AutoReportPlanOutcome::Count ||
      abi_plan.reason == AutoReportPlanReason::Count) {
    return false;
  }
  if (plan_auto_report(sizing_inventory, abi_plan.ceiling_bytes) != abi_plan)
    return false;
  if (abi_plan.complete())
    return abi_plan.reason == AutoReportPlanReason::None && abi_plan.layout.valid &&
           abi_plan.layout.required_bytes == abi_plan.required_bytes;
  return abi_plan.reason != AutoReportPlanReason::None && !abi_plan.layout.valid;
}

EvidenceRequirements detail::plan_evidence_requirements(const EvidencePlanningContext &context) {
  ReportRequirements requirements;
  requirements.reason = detail::validate_evidence_intents(context.observation_plan, Mode::Default);
  if (requirements.reason != EvidenceRequirementReason::None)
    return requirements;

  AutoReportInventory inventory;
  (void)detail::accumulate_evidence_counts(context.observation_plan,
                                           context.maximum_access_probe_count, inventory);

  const uint64_t banks_per_range = context.watchpoint_banks == 0 ? 8u : context.watchpoint_banks;
  const uint64_t access_banks = util::saturating_mul(inventory.access_range_count, banks_per_range);
  inventory.range_bank_count = access_banks;
  const uint64_t slots =
      access_banks == 0u ? 0u : util::saturating_add(access_banks, inventory.atomic_event_count);
  inventory.sync_slot_count = slots;
  inventory.watchpoint_count = slots;
  inventory.bank_count_adaptive = inventory.access_range_count != 0u;
  // ConSan diagnoses on the host; its optional immediate checker only
  // increments event_counter. No device diagnostic records are published.
  // Reclaim this unused reservation to pay for exact masks without losing banks.
  inventory = fit_auto_report_inventory(inventory, context.requested_report_buffer_size);

  detail::publish_evidence_requirements(requirements, std::move(inventory),
                                        context.requested_report_buffer_size);
  return requirements;
}

} // namespace rocjitsu::consan
