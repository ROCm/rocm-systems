// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_evidence_planning.h
/// @brief Forward-only projection of admitted evidence into lowering sites.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

/// Immutable projection of the one decoded program container that owns an
/// admitted evidence site. Kernel symbol fallback, runtime-kernel exclusion,
/// and descriptor provenance are resolved once at this boundary rather than
/// being rediscovered by each evidence kind.
struct MoiEvidenceContainerView {
  std::string qualified_name;
  const ProgramInventory *inventory = nullptr;
  ConSanProgramContainerRef container;
  std::optional<uint64_t> kernel_descriptor_file_offset;
  uint64_t entry_text_offset = 0, text_file_offset = 0, code_size = 0;
  bool uses_cluster_workgroup_id = false;

  template <typename Site> [[nodiscard]] const Site *find_site(uint64_t text_offset) const {
    return inventory == nullptr ? nullptr
                                : inventory->find_unique_decoded_site<Site>(container, text_offset);
  }
};

[[nodiscard]] std::optional<MoiEvidenceContainerView>
resolve_moi_evidence_container(const ProgramInventory &inventory, bool in_kernel,
                               std::string_view container_name,
                               std::span<const ConSanExecutionOwner> execution_owners);

[[nodiscard]] std::vector<consan_detail::MoiBarrierEvidenceSitePlan>
build_moi_barrier_evidence_site_plans(const ProgramInventory &inventory,
                                      const ConSanObservationPlan &observation,
                                      ConSanProbeIntentKind evidence_kind,
                                      std::vector<std::string> &errors);

[[nodiscard]] std::vector<consan_detail::MoiFenceEvidenceSitePlan>
build_moi_fence_evidence_site_plans(const ProgramInventory &inventory,
                                    const ConSanObservationPlan &observation,
                                    std::vector<std::string> &errors);

[[nodiscard]] std::vector<consan_detail::MoiAtomicEvidenceSitePlan>
build_moi_atomic_evidence_site_plans(const ProgramInventory &inventory,
                                     const ConSanObservationPlan &observation,
                                     ConSanProbeIntentKind evidence_kind,
                                     std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
