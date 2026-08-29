// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

namespace rocjitsu {

/// Deliberately narrow, immutable projection for verbose patch diagnostics.
/// It contains presentation facts only, never validation identities, relay
/// chains, mutation proof, or owner/ABI authority.
struct ConSanPatchDebugRecord {
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

/// Read-only diagnostic projection of lowerer-private proof artifacts.
///
/// This internal opt-in interface supports verbose development logs and
/// invariant tests. It is deliberately absent from TransformResult's public
/// production interface: no install, attribution, coverage, or retry decision
/// may depend on this projection.
struct ConSanTransformDebugReport {
  std::span<const ConSanFaultSite> fault_sites;
  std::span<const ConSanBarrierMoveDestination> barrier_move_destinations;
  std::span<const ConSanFaultMutationPlan> fault_plans;
  std::span<const ConSanCandidateResourcePlan> resource_plans;
  std::span<const ConSanCommittedLowering> committed_lowerings;
  ConSanResourcePlanSummary resource_summary;
  std::vector<ConSanPatchDebugRecord> patches;
};

[[nodiscard]] ConSanTransformDebugReport
consan_transform_debug_report(const TransformResult &result);

} // namespace rocjitsu
