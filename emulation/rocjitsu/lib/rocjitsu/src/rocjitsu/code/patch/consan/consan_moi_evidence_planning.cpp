// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu::consan_moi_impl {

const ConSanProgramContainer *resolve_moi_evidence_container(const ProgramInventory &inventory,
                                                             ConSanProgramSiteId source_site) {
  const ConSanProgramSite *site = inventory.program_site(source_site);
  const ConSanProgramContainer *container =
      site == nullptr ? nullptr : inventory.container(site->container);
  if (site == nullptr || container == nullptr ||
      (container->is_kernel() && is_rocclr_runtime_kernel_name(container->name))) {
    return nullptr;
  }
  return container;
}

std::string moi_evidence_container_name(const ConSanProgramContainer &container) {
  return std::string(container.is_kernel() ? "kernel:" : "function:") + container.name;
}

std::string moi_evidence_container_name(const ProgramInventory &inventory,
                                        ConSanProgramSiteId source_site) {
  const ConSanProgramContainer *container = resolve_moi_evidence_container(inventory, source_site);
  return container == nullptr ? "<invalid-container>" : moi_evidence_container_name(*container);
}

std::optional<ConSanAtomicSite>
materialize_moi_communication_site(const ProgramInventory &inventory,
                                   ConSanProgramSiteId source_site,
                                   ConSanSyncSequenceId sequence_id) {
  const ConSanSyncSequence *sequence = inventory.sync().find_sequence(sequence_id);
  const ConSanProgramSite *source = inventory.program_site(source_site);
  if (sequence == nullptr || source == nullptr)
    return std::nullopt;

  ConSanAtomicSite result;
  if (const ConSanAtomicSite *atomic = source->get_if<ConSanAtomicSite>()) {
    result = *atomic;
  } else if (const ConSanOrdinaryMemorySite *ordinary = source->get_if<ConSanOrdinaryMemorySite>();
             ordinary != nullptr &&
             (ordinary->support_reason == ConSanOrdinaryMemorySupportReason::Supported ||
              ordinary->support_reason ==
                  ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly)) {
    result = consan_atomic_communication_site(*ordinary);
  } else {
    return std::nullopt;
  }
  if (sequence->scope)
    result.scope = sequence->scope;
  if (result.size == 0u || result.width_bits == 0u)
    return std::nullopt;
  return result;
}

std::optional<ConSanMoiAtomicEventKind> moi_atomic_event_kind(ConSanSyncMemoryRole role) {
  switch (role) {
  case ConSanSyncMemoryRole::Release:
    return ConSanMoiAtomicEventKind::Release;
  case ConSanSyncMemoryRole::Acquire:
    return ConSanMoiAtomicEventKind::Acquire;
  case ConSanSyncMemoryRole::AcquireRelease:
    return ConSanMoiAtomicEventKind::AcquireRelease;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<MoiAtomicEvidenceSourceView>
resolve_moi_atomic_evidence_source(const ProgramInventory &inventory,
                                   const consan_detail::MoiAtomicEvidenceSitePlan &plan) {
  const SynchronizationInventoryView graph = inventory.sync();
  const ConSanSyncEvent *event = graph.find_event(plan.event);
  const ConSanSyncSequence *sequence = graph.find_sequence(plan.sequence);
  const std::optional<ConSanAtomicSite> site =
      materialize_moi_communication_site(inventory, plan.source_site, plan.sequence);
  if (event == nullptr || sequence == nullptr || !site ||
      resolve_moi_evidence_container(inventory, plan.source_site) == nullptr ||
      event->source_site != plan.source_site ||
      (event->kind != ConSanSyncKind::Atomic && event->kind != ConSanSyncKind::OrdinaryMemory) ||
      graph.find_unique_sequence_containing(event->semantic_id) != sequence ||
      !moi_atomic_event_kind(sequence->memory_role) ||
      sequence->end_text_offset < site->text_offset + site->size) {
    return std::nullopt;
  }
  return MoiAtomicEvidenceSourceView{event, sequence, *site};
}

std::optional<MoiFenceEvidenceSourceView>
resolve_moi_fence_evidence_source(const ProgramInventory &inventory,
                                  const consan_detail::MoiFenceEvidenceSitePlan &plan) {
  const SynchronizationInventoryView graph = inventory.sync();
  const ConSanSyncEvent *fence_event = graph.find_event(plan.event);
  const ConSanSyncSequence *sequence = graph.find_sequence(plan.sequence);
  if (fence_event == nullptr || sequence == nullptr || fence_event->kind != ConSanSyncKind::Fence ||
      resolve_moi_evidence_container(inventory, fence_event->source_site) == nullptr ||
      resolve_moi_evidence_container(inventory, plan.source_site) == nullptr) {
    return std::nullopt;
  }

  const ConSanMoiFenceCandidate *association = nullptr;
  for (const ConSanMoiFenceCandidate &candidate : graph.moi_fence_candidates) {
    if (candidate.fence_event != plan.event || candidate.sequence != plan.sequence ||
        !candidate.communication_event || !candidate.eligible()) {
      continue;
    }
    const ConSanSyncEvent *communication = graph.find_event(*candidate.communication_event);
    if (communication == nullptr || communication->source_site != plan.source_site)
      continue;
    if (association != nullptr)
      return std::nullopt;
    association = &candidate;
  }
  const ConSanSyncEvent *communication_event =
      association == nullptr ? nullptr : graph.find_event(*association->communication_event);
  const ConSanFenceSite *fence_site =
      inventory.program_site<ConSanFenceSite>(fence_event->source_site);
  const std::optional<ConSanAtomicSite> communication_site =
      materialize_moi_communication_site(inventory, plan.source_site, plan.sequence);
  if (association == nullptr || communication_event == nullptr || fence_site == nullptr ||
      !communication_site ||
      (association->memory_role != ConSanSyncMemoryRole::Release &&
       association->memory_role != ConSanSyncMemoryRole::Acquire) ||
      (communication_event->kind != ConSanSyncKind::Atomic &&
       communication_event->kind != ConSanSyncKind::OrdinaryMemory) ||
      !graph.same_container(*communication_event, *fence_event) ||
      graph.execution_owners(*communication_event).empty()) {
    return std::nullopt;
  }

  MoiFenceEvidenceSourceView result{association, fence_event, communication_event,
                                    sequence,    fence_site,  *communication_site};
  if (sequence->acquire_polling_loop_header_text_offset) {
    const uint64_t loop_header = *sequence->acquire_polling_loop_header_text_offset;
    if (sequence->kind != ConSanSyncKind::OrdinaryMemory ||
        sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
        communication_event->kind != ConSanSyncKind::OrdinaryMemory ||
        loop_header >= sequence->begin_text_offset || sequence->end_text_offset <= loop_header ||
        sequence->begin_text_offset - loop_header > communication_site->file_offset ||
        sequence->end_text_offset - loop_header > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
  }
  if (result.captures_address_before_guest() &&
      (sequence->kind != ConSanSyncKind::OrdinaryMemory ||
       sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
       sequence->begin_text_offset != communication_site->text_offset ||
       sequence->end_text_offset <= sequence->begin_text_offset ||
       sequence->end_text_offset < fence_event->text_offset() + fence_site->size ||
       sequence->end_text_offset - sequence->begin_text_offset >
           std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  if (result.patch_size() == 0u ||
      result.patch_text_offset() > std::numeric_limits<uint64_t>::max() - result.patch_size()) {
    return std::nullopt;
  }
  return result;
}

} // namespace rocjitsu::consan_moi_impl
