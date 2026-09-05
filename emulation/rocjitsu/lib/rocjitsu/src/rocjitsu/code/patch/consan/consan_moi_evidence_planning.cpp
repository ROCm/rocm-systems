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

} // namespace rocjitsu::consan_moi_impl
