// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_inline_shadow.h
/// @brief Compiled InlineShadow engine lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

/// Exact InlineShadow scalar facts projected at its mode-planning boundary.
/// Resource sizing, entry prologues, and body emission share this value
/// without inspecting the complete mutable operating point.
struct MoiInlineShadowScalarState {
  std::optional<uint16_t> exec_save_sgpr;
  std::optional<ConSanMoiRouterCallSgprs> router_call;
  std::optional<uint16_t> visible_evidence_sgpr;
  bool inline_scalar_spill = false;
  bool scalar_spill = false;
  bool branch_only_spill = false;
  bool exec_save_persistent = false;
  bool dynamic_stack_spill = false;
};

[[nodiscard]] MoiInlineShadowScalarState
project_inline_shadow_scalar_state(const ConSanMoiOperatingPoint &point);

[[nodiscard]] uint16_t inline_shadow_loop_scratch_count(const ConSanMoiCandidate &candidate);
[[nodiscard]] uint16_t inline_shadow_scratch_count(bool track_atomics,
                                                   const MoiAccessResourceFacts &resource_facts,
                                                   const ConSanMoiCandidate &candidate);
[[nodiscard]] uint16_t
inline_shadow_spill_backed_scratch_count(bool track_atomics,
                                         const MoiAccessResourceFacts &resource_facts,
                                         const ConSanMoiCandidate &candidate);

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const MoiInlineShadowScalarState &state,
                                                         bool inline_access_present,
                                                         uint16_t required_sgpr_count,
                                                         rj_code_arch_t arch,
                                                         std::vector<std::string> &errors);

[[nodiscard]] std::optional<uint16_t>
inline_shadow_visible_evidence_sgpr(const MoiInlineShadowScalarState &state);

void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                   std::span<const ConSanMoiCandidate> admitted,
                                   const MoiObjectModeSemantics &semantics,
                                   ConSanTransformArtifacts &result);

void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            rj_code_arch_t arch,
                                            const MoiObjectModeSemantics &semantics,
                                            ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
