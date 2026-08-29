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

namespace rocjitsu {

/// Presentation-only snapshot of an injectable instruction.
struct ConSanFaultSiteDiagnostic {
  ConSanFaultSiteKind kind = ConSanFaultSiteKind::Barrier;
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t occurrence = 0;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::string mnemonic;
  std::string semantic_role;
  std::string decoded_operands;
  ConSanOrdinaryMemorySupportReason ordinary_memory_support_reason =
      ConSanOrdinaryMemorySupportReason::NotApplicable;
  std::optional<std::string> sync_event_identity;
  std::optional<std::string> sync_sequence_identity;
  ConSanSemanticConfidence sync_confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSyncMemoryRole sync_memory_role = ConSanSyncMemoryRole::Unknown;
  std::vector<ConSanExecutionOwner> execution_owners;
};

/// Presentation-only snapshot of a barrier-movement destination decision.
struct ConSanBarrierMoveDestinationDiagnostic {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::string mnemonic;
  bool memory_operation = false;
  ConSanBarrierMoveDestinationIssue issue = ConSanBarrierMoveDestinationIssue::None;
  std::string issue_detail;
  ConSanBarrierMoveCfgContract cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_source_offset;
  std::vector<ConSanExecutionOwner> execution_owners;

  [[nodiscard]] bool suitable() const { return issue == ConSanBarrierMoveDestinationIssue::None; }
};

/// Presentation-only snapshot of a selected validation mutation.
struct ConSanFaultMutationDiagnostic {
  ConSanFaultMutationKind kind = ConSanFaultMutationKind::DropBarrier;
  std::string primary_identity;
  std::optional<std::string> companion_identity;
  std::optional<std::string> logical_sequence_identity;
  std::vector<std::string> ordered_member_identities;
  std::optional<std::string> destination_identity;
  ConSanBarrierMoveDirection barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker;
  ConSanBarrierMoveCfgContract barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  std::optional<int32_t> original_barrier_id;
  std::optional<int32_t> target_barrier_id;
  ConSanBarrierSite::Scope original_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  ConSanBarrierSite::Scope target_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  std::optional<uint16_t> target_address_vgpr;
};

/// Aggregated presentation facts for one resource-rejection class.
struct ConSanResourceFailureDiagnostic {
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  ConSanRegisterPlanReason reason = ConSanRegisterPlanReason::None;
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
struct ConSanResourceAlternativeDiagnostic {
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  size_t candidate_index = 0;
  uint64_t text_offset = 0;
  size_t attempt_index = 0;
  ConSanResourcePlanAlternativeKind kind =
      ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill;
  uint16_t scratch_vgpr_count = 0;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
  ConSanRegisterPlanReason reason = ConSanRegisterPlanReason::None;
  ConSanResourcePlanAlternativeOutcome outcome = ConSanResourcePlanAlternativeOutcome::Rejected;
};

/// Presentation-only snapshot of emitted patch mechanics.
struct ConSanPatchDiagnostic {
  ConSanPatchKind kind = ConSanPatchKind::InlineNopRewrite;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  uint32_t trampoline_size = 0;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> scalar_vcc_spill_sgpr;
  std::optional<uint16_t> scalar_vcc_spill_vgpr;
  uint16_t scalar_vcc_spill_vgpr_count = 0;
  std::optional<uint32_t> persistent_epoch_private_offset;
  uint16_t spilled_vgpr_count = 0;
  uint32_t required_private_segment_size = 0;
  uint32_t dynamic_private_segment_addend = 0;
  uint32_t workgroup_shadow_base = 0;
  uint32_t workgroup_shadow_size = 0;
  uint32_t required_group_segment_size = 0;
  uint32_t sampled_first_slot = 0;
  uint32_t sampled_window_bank_count = 0;
  ConSanLdsAccessKind sampled_access_kind = ConSanLdsAccessKind::Other;
};

/// Owned diagnostic projection of a completed transform.
///
/// No member aliases lowerer-private storage. Installation, coverage, runtime
/// mapping, retry, and validation decisions must use TransformResult's typed
/// production contracts instead.
struct ConSanTransformDiagnosticReport {
  std::vector<ConSanFaultSiteDiagnostic> fault_sites;
  std::vector<ConSanBarrierMoveDestinationDiagnostic> barrier_move_destinations;
  std::vector<ConSanFaultMutationDiagnostic> fault_mutations;
  ConSanResourcePlanSummary resource_summary;
  std::vector<ConSanResourceFailureDiagnostic> resource_failures;
  std::vector<ConSanResourceAlternativeDiagnostic> resource_alternatives;
  std::vector<ConSanPatchDiagnostic> patches;
};

[[nodiscard]] ConSanTransformDiagnosticReport
consan_transform_diagnostic_report(const TransformResult &result);

} // namespace rocjitsu
