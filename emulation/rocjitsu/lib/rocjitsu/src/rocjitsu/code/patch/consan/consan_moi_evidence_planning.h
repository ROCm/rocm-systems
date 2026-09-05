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

/// Reconstruct the operand-rich communication view at a native-lowering
/// boundary from authoritative inventory handles. Ordered ordinary-memory
/// operations are normalized to the same transient view as native atomics;
/// sequence-qualified scope supersedes the decoder's instruction-local fact.
[[nodiscard]] std::optional<ConSanAtomicSite>
materialize_moi_communication_site(const ProgramInventory &inventory,
                                   ConSanProgramSiteId source_site,
                                   ConSanSyncSequenceId sequence);

/// Short-lived lowering view resolved from an atomic evidence plan. Pointers
/// refer into the immutable inventory that owns the plan's handles; decoded
/// operands are materialized only in this view.
struct MoiAtomicEvidenceSourceView {
  const ConSanSyncEvent *event = nullptr;
  const ConSanSyncSequence *sequence = nullptr;
  ConSanAtomicSite site;

  [[nodiscard]] bool is_rmw() const {
    return event != nullptr && event->kind == ConSanSyncKind::Atomic;
  }
};

[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
moi_atomic_event_kind(ConSanSyncMemoryRole role);

[[nodiscard]] std::optional<MoiAtomicEvidenceSourceView>
resolve_moi_atomic_evidence_source(const ProgramInventory &inventory,
                                   const consan_detail::MoiAtomicEvidenceSitePlan &plan);

/// Short-lived Record/Replay view of one qualified fence association. All
/// pointers refer into the immutable inventory; replacement geometry is
/// computed from those canonical facts instead of being cached in the plan.
struct MoiFenceEvidenceSourceView {
  const ConSanMoiFenceCandidate *association = nullptr;
  const ConSanSyncEvent *fence_event = nullptr;
  const ConSanSyncEvent *communication_event = nullptr;
  const ConSanSyncSequence *sequence = nullptr;
  const ConSanFenceSite *fence_site = nullptr;
  ConSanAtomicSite communication_site;

  [[nodiscard]] bool is_resolved() const {
    return association != nullptr && fence_event != nullptr && communication_event != nullptr &&
           sequence != nullptr && fence_site != nullptr && communication_site.size != 0u &&
           communication_site.width_bits != 0u;
  }
  [[nodiscard]] bool captures_address_before_guest() const {
    return association->memory_role == ConSanSyncMemoryRole::Acquire &&
           communication_event->kind == ConSanSyncKind::OrdinaryMemory;
  }
  [[nodiscard]] uint64_t patch_text_offset() const {
    return captures_address_before_guest() ? sequence->begin_text_offset
                                           : fence_event->text_offset();
  }
  [[nodiscard]] uint64_t patch_file_offset() const {
    return captures_address_before_guest() ? communication_site.file_offset
                                           : fence_site->file_offset;
  }
  [[nodiscard]] uint32_t patch_size() const {
    return captures_address_before_guest()
               ? static_cast<uint32_t>(sequence->end_text_offset - sequence->begin_text_offset)
               : fence_site->size;
  }
  [[nodiscard]] std::optional<uint64_t> scalar_clause_text_offset() const {
    return captures_address_before_guest() ? sequence->scalar_clause_text_offset : std::nullopt;
  }
};

[[nodiscard]] std::optional<MoiFenceEvidenceSourceView>
resolve_moi_fence_evidence_source(const ProgramInventory &inventory,
                                  const consan_detail::MoiFenceEvidenceSitePlan &plan);

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
