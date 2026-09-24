// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_evidence_planning.h
/// @brief Forward-only projection of admitted evidence into lowering sites.

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"

namespace rocjitsu::consan::detail {

/// Resolve an evidence source's authoritative container through its stable
/// arena handle. Runtime kernels and stale or inconsistent handles fail closed.
[[nodiscard]] const ProgramContainer *resolve_evidence_container(const ProgramInventory &inventory,
                                                                 ProgramSiteId source_site);

/// Materialize the qualified spelling only at a diagnostic boundary.
[[nodiscard]] std::string evidence_container_name(const ProgramContainer &container);
[[nodiscard]] std::string evidence_container_name(const ProgramInventory &inventory,
                                                  ProgramSiteId source_site);

/// Reconstruct the operand-rich communication view at a native-lowering
/// boundary from authoritative inventory handles. Ordered ordinary-memory
/// operations are normalized to the same transient view as native atomics;
/// sequence-qualified scope supersedes the decoder's instruction-local fact.
[[nodiscard]] std::optional<AtomicSite>
materialize_communication_site(const ProgramInventory &inventory, ProgramSiteId source_site,
                               SyncSequenceId sequence);

/// Short-lived lowering view resolved from an atomic evidence plan. Pointers
/// refer into the immutable inventory that owns the plan's handles; decoded
/// operands are materialized only in this view.
struct AtomicEvidenceSourceView {
  const SyncEvent *event = nullptr;
  const SyncSequence *sequence = nullptr;
  AtomicSite site;
  bool native_modification = false;

  [[nodiscard]] bool is_rmw() const {
    return event ? event->kind == SyncKind::Atomic : native_modification;
  }
  [[nodiscard]] bool relocates_polling_loop() const {
    return sequence && sequence->acquire_polling_loop_header_text_offset &&
           *sequence->acquire_polling_loop_header_text_offset < sequence->begin_text_offset;
  }
  [[nodiscard]] uint64_t patch_text_offset() const {
    return relocates_polling_loop() ? *sequence->acquire_polling_loop_header_text_offset
                                    : site.text_offset;
  }
  [[nodiscard]] uint64_t patch_file_offset() const {
    return relocates_polling_loop() ? site.file_offset - (site.text_offset - patch_text_offset())
                                    : site.file_offset;
  }
};

/// Coverage of decoded modifications in every kernel owning a selected capture.
/// This checks original sites against the actual lowering plans, including
/// unsupported/excluded sites that contributed no observation intent.
struct PublicationModificationCoverage {
  uint32_t required = 0;
  uint32_t covered = 0;
  bool valid = false;
  std::vector<ProgramSiteId> missing;

  [[nodiscard]] bool complete() const { return valid && required != 0u && missing.empty(); }
};

[[nodiscard]] PublicationModificationCoverage
publication_modification_coverage(const ProgramInventory &inventory,
                                  std::span<const AtomicEvidenceSitePlan> captures);

[[nodiscard]] std::optional<AtomicEventKind> atomic_event_kind(SyncMemoryRole role);

[[nodiscard]] std::optional<AtomicEvidenceSourceView>
resolve_atomic_evidence_source(const ProgramInventory &inventory,
                               const detail::AtomicEvidenceSitePlan &plan);

[[nodiscard]] std::vector<detail::BarrierEvidenceSitePlan>
build_barrier_evidence_site_plans(const ProgramInventory &inventory,
                                  const ObservationPlan &observation, ProbeIntentKind evidence_kind,
                                  std::vector<std::string> &errors);

[[nodiscard]] std::vector<detail::AtomicEvidenceSitePlan>
build_atomic_evidence_site_plans(const ProgramInventory &inventory,
                                 const ObservationPlan &observation, ProbeIntentKind evidence_kind,
                                 std::vector<std::string> &errors);

} // namespace rocjitsu::consan::detail
