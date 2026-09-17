// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_transform_diagnostics.h
/// @brief Owned presentation facts from a completed ConSan transform.

#pragma once

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::consan {

/// Presentation-only snapshots copied from the immutable facets of completed
/// analysis products. These aliases cannot expose the private source binding,
/// target operands, or application proof carried by the derived products.
struct FaultSiteDiagnostic : FaultSitePresentation {
  std::optional<std::string> sync_event_identity;
  std::optional<std::string> sync_sequence_identity;
};
using BarrierMoveDestinationDiagnostic = BarrierMoveDestinationPresentation;
using FaultMutationDiagnostic = FaultMutationPresentation;

/// Aggregated presentation facts for one resource-rejection class.
struct ResourceFailureDiagnostic {
  ResourceSiteKind site_kind = ResourceSiteKind::Access;
  RegisterPlanReason reason = RegisterPlanReason::None;
  size_t count = 0;
  uint16_t min_scratch_vgprs = 0;
  uint16_t max_scratch_vgprs = 0;
  uint16_t min_current_vgprs = 0;
  uint16_t max_current_vgprs = 0;
  uint16_t min_max_referenced_vgprs = 0;
  uint16_t max_max_referenced_vgprs = 0;
  uint16_t min_ordinary_vgpr_limit = 0;
  uint16_t max_ordinary_vgpr_limit = 0;
  uint16_t min_required_vgprs = 0;
  uint16_t max_required_vgprs = 0;
  size_t min_owners = 0;
  size_t max_owners = 0;
  bool has_indirect_vgpr_access = false;
};

/// Flattened presentation facts for one resource-planner alternative.
struct ResourceAlternativeDiagnostic {
  ResourceSiteKind site_kind = ResourceSiteKind::Access;
  size_t candidate_index = 0;
  uint64_t text_offset = 0;
  size_t attempt_index = 0;
  ResourcePlanAlternativeKind kind = ResourcePlanAlternativeKind::GuestOperandOverlapSpill;
  uint16_t scratch_vgpr_count = 0;
  RegisterAllocationSource source = RegisterAllocationSource::Unsupported;
  RegisterPlanReason reason = RegisterPlanReason::None;
  ResourcePlanAlternativeOutcome outcome = ResourcePlanAlternativeOutcome::Rejected;
};

/// Presentation-only snapshot of emitted patch mechanics.
struct PatchDiagnostic {
  std::string kind;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  uint32_t trampoline_size = 0;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<ScalarVccSpill> scalar_vcc_spill;
  std::optional<uint32_t> persistent_epoch_private_offset;
  uint16_t spilled_vgpr_count = 0;
  uint32_t required_private_segment_size = 0;
  uint32_t dynamic_private_segment_addend = 0;
};

/// Owned diagnostic projection of a completed transform.
///
/// No member aliases lowerer-private storage. Installation, coverage, runtime
/// mapping, retry, and validation decisions must use TransformResult's typed
/// production contracts instead.
struct TransformDiagnosticReport {
  std::vector<FaultSiteDiagnostic> fault_sites;
  std::vector<BarrierMoveDestinationDiagnostic> barrier_move_destinations;
  std::vector<FaultMutationDiagnostic> fault_mutations;
  ResourcePlanSummary resource_summary;
  std::vector<ResourceFailureDiagnostic> resource_failures;
  std::vector<ResourceAlternativeDiagnostic> resource_alternatives;
  std::vector<PatchDiagnostic> patches;
};

/// Materialize one owned presentation snapshot from the authoritative
/// program and synchronization inventories.
[[nodiscard]] std::optional<FaultSiteDiagnostic>
fault_site_diagnostic(const ProgramInventory &inventory, const FaultSite &site);

[[nodiscard]] TransformDiagnosticReport transform_diagnostic_report(const TransformResult &result);

} // namespace rocjitsu::consan
