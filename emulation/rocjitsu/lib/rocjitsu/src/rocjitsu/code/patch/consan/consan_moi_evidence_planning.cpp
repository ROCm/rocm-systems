// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"

#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"

namespace rocjitsu::consan_moi_impl {

const ConSanProgramContainer *resolve_moi_evidence_container(const ProgramInventory &inventory,
                                                             ConSanProgramSiteId source_site) {
  const ConSanProgramSite *site = inventory.program_site(source_site);
  const ConSanProgramContainer *container =
      site == nullptr ? nullptr : inventory.container(site->container.id);
  if (site == nullptr || container == nullptr || site->container.kind != container->kind ||
      site->container.name != container->name ||
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
  } else if (const ConSanOrdinaryMemorySite *ordinary =
                 source->get_if<ConSanOrdinaryMemorySite>();
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

} // namespace rocjitsu::consan_moi_impl
