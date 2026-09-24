// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_evidence_planning.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

#include <algorithm>
#include <set>

namespace rocjitsu::consan::detail {

const ProgramContainer *resolve_evidence_container(const ProgramInventory &inventory,
                                                   ProgramSiteId source_site) {
  const ProgramSite *site = inventory.program_site(source_site);
  const ProgramContainer *container =
      site == nullptr ? nullptr : inventory.container(site->container);
  if (site == nullptr || container == nullptr ||
      (container->is_kernel() && is_rocclr_runtime_kernel_name(container->name))) {
    return nullptr;
  }
  return container;
}

std::string evidence_container_name(const ProgramContainer &container) {
  return std::string(container.is_kernel() ? "kernel:" : "function:") + container.name;
}

std::string evidence_container_name(const ProgramInventory &inventory, ProgramSiteId source_site) {
  const ProgramContainer *container = resolve_evidence_container(inventory, source_site);
  return container == nullptr ? "<invalid-container>" : evidence_container_name(*container);
}

std::optional<AtomicSite> materialize_communication_site(const ProgramInventory &inventory,
                                                         ProgramSiteId source_site,
                                                         SyncSequenceId sequence_id) {
  const SyncSequence *sequence = inventory.sync().find_sequence(sequence_id);
  const ProgramSite *source = inventory.program_site(source_site);
  if (sequence == nullptr || source == nullptr)
    return std::nullopt;

  AtomicSite result;
  if (const AtomicSite *atomic = source->get_if<AtomicSite>()) {
    result = *atomic;
  } else if (const OrdinaryMemorySite *ordinary = source->get_if<OrdinaryMemorySite>();
             ordinary != nullptr &&
             (ordinary->support_reason == OrdinaryMemorySupportReason::Supported ||
              ordinary->support_reason ==
                  OrdinaryMemorySupportReason::SupportedSynchronizationOnly)) {
    result = atomic_communication_site(*ordinary);
  } else {
    return std::nullopt;
  }
  if (sequence->scope)
    result.scope = sequence->scope;
  if (result.size == 0u || result.width_bits == 0u)
    return std::nullopt;
  return result;
}

std::optional<AtomicEventKind> atomic_event_kind(SyncMemoryRole role) {
  switch (role) {
  case SyncMemoryRole::Release:
    return AtomicEventKind::Release;
  case SyncMemoryRole::Acquire:
    return AtomicEventKind::Acquire;
  case SyncMemoryRole::AcquireRelease:
    return AtomicEventKind::AcquireRelease;
  case SyncMemoryRole::Unknown:
  case SyncMemoryRole::None:
  case SyncMemoryRole::SequentiallyConsistent:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<AtomicEvidenceSourceView>
resolve_atomic_evidence_source(const ProgramInventory &inventory,
                               const detail::AtomicEvidenceSitePlan &plan) {
  const SynchronizationInventoryView graph = inventory.sync();
  if (plan.publication_modification) {
    const ProgramSite *source = inventory.program_site(plan.source_site);
    if (!source || !resolve_evidence_container(inventory, plan.source_site))
      return std::nullopt;
    if (const auto *atomic = source->get_if<AtomicSite>())
      return AtomicEvidenceSourceView{nullptr, nullptr, *atomic, true};
    if (const auto *ordinary = source->get_if<OrdinaryMemorySite>();
        ordinary && ordinary->operation == OrdinaryMemoryOperation::Store)
      return AtomicEvidenceSourceView{nullptr, nullptr, atomic_communication_site(*ordinary),
                                      false};
    return std::nullopt;
  }
  const SyncEvent *event = graph.find_event(plan.event);
  const SyncSequence *sequence = graph.find_sequence(plan.sequence);
  const std::optional<AtomicSite> site =
      materialize_communication_site(inventory, plan.source_site, plan.sequence);
  if (event == nullptr || sequence == nullptr || !site ||
      resolve_evidence_container(inventory, plan.source_site) == nullptr ||
      event->source_site != plan.source_site ||
      (event->kind != SyncKind::Atomic && event->kind != SyncKind::OrdinaryMemory) ||
      graph.find_unique_sequence_containing(event->semantic_id) != sequence ||
      !atomic_event_kind(sequence->memory_role) ||
      sequence->end_text_offset < site->text_offset + site->size) {
    return std::nullopt;
  }
  AtomicEvidenceSourceView result{event, sequence, *site};
  if (sequence->acquire_polling_loop_header_text_offset) {
    const uint64_t loop_header = *sequence->acquire_polling_loop_header_text_offset;
    if (sequence->kind != SyncKind::OrdinaryMemory ||
        sequence->memory_role != SyncMemoryRole::Acquire ||
        event->kind != SyncKind::OrdinaryMemory || loop_header >= sequence->begin_text_offset ||
        sequence->begin_text_offset != site->text_offset ||
        sequence->end_text_offset <= loop_header ||
        site->text_offset - loop_header > site->file_offset ||
        sequence->end_text_offset - loop_header > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
  }
  return result;
}

PublicationModificationCoverage
publication_modification_coverage(const ProgramInventory &inventory,
                                  std::span<const AtomicEvidenceSitePlan> captures) {
  PublicationModificationCoverage result;
  std::set<uint64_t> owners;
  for (const auto &capture : captures) {
    if (!capture.is_well_formed() || !resolve_atomic_evidence_source(inventory, capture))
      return result;
    const auto descriptors = inventory.execution_owner_descriptors(capture.source_site);
    if (descriptors.empty())
      return result;
    owners.insert(descriptors.begin(), descriptors.end());
  }
  if (owners.empty())
    return result;
  result.valid = true;
  for (const auto &site : inventory.program_sites()) {
    const auto *atomic = site.get_if<AtomicSite>();
    const auto *ordinary = site.get_if<OrdinaryMemorySite>();
    if ((!atomic && (!ordinary || ordinary->operation != OrdinaryMemoryOperation::Store)) ||
        site.mnemonic_view().starts_with("ds_"))
      continue;
    const auto descriptors = inventory.execution_owner_descriptors(site);
    if (!std::ranges::any_of(descriptors, [&](uint64_t owner) { return owners.contains(owner); }))
      continue;
    ++result.required;
    const bool covered = std::ranges::any_of(captures, [&](const AtomicEvidenceSitePlan &capture) {
      const auto *source = inventory.program_site(capture.source_site);
      // Physical aliases only share a capture if their operand-rich decodes
      // agree. Matching offsets alone could hide an unsupported alias.
      return source && source->physical_id == site.physical_id && source->same_payload(site);
    });
    if (covered)
      ++result.covered;
    else
      result.missing.push_back(site.id);
  }
  return result;
}

} // namespace rocjitsu::consan::detail
