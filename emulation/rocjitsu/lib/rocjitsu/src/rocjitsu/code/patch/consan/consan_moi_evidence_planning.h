// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_evidence_planning.h
/// @brief Forward-only projection of admitted evidence into lowering sites.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

/// Resolve an evidence source's authoritative container through its stable
/// arena handle. Runtime kernels and stale or inconsistent handles fail closed.
[[nodiscard]] const ConSanProgramContainer *
resolve_moi_evidence_container(const ProgramInventory &inventory, ConSanProgramSiteId source_site);

/// Materialize the qualified spelling only at a diagnostic boundary.
[[nodiscard]] std::string moi_evidence_container_name(const ConSanProgramContainer &container);
[[nodiscard]] std::string moi_evidence_container_name(const ProgramInventory &inventory,
                                                      ConSanProgramSiteId source_site);

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
