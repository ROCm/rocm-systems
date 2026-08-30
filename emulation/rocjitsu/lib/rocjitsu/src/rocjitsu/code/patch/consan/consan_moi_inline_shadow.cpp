// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// InlineShadow access and atomic-order lowering share one engine-private
// translation unit. Their target-neutral entry points are the only surface
// visible to MOI orchestration; exact-shadow representation and route choices
// remain owned by this engine component.

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_local_island_allocator.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::append_moi_workitem_owner_derivation;
using consan_detail::build_moi_relocated_guest_access_words;
using consan_detail::moi_guest_access_relocation_requires_adjusted_address;
using consan_detail::moi_workgroup_shadow_initialization_lanes;
using consan_detail::moi_workgroup_shadow_preferred_zero_vgpr_count;
using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkgroupShadowClearStoreForm;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::plan_moi_workgroup_shadow_clear;
using consan_detail::range_overlaps;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_moi_scc_preserving_indirect_jump;
using consan_moi_detail::append_publish_visible_evidence_if_zero;
using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;
using consan_moi_detail::count_nop_padding;
using consan_moi_detail::decode_relocatable_entry_instruction;
using consan_moi_detail::moi_has_runtime_hardware_dispatch_id;
using consan_moi_detail::note_moi_persistent_vgpr_state;
using consan_moi_detail::resolve_moi_report_layout;

namespace consan_moi_impl {

MoiObjectModePlan plan_inline_shadow_object_mode(const ConSanRequest &request,
                                                 const ConSanMoiOperatingPoint &point,
                                                 const MoiObjectFacts &facts,
                                                 const ConSanObservationPlan &observation_plan) {
  MoiObjectModePlan plan = make_moi_object_mode_plan(request, point, ConSanMoiOwnerSource::HwId);
  plan.prologue_requires_consumer = true;
  if (plan.owner_source == ConSanMoiOwnerSource::WorkitemId) {
    plan.errors.emplace_back(
        "ConSan MOI Inline Shadow requires resident-wave ownership; workitem_id_x is not exact "
        "for multidimensional workgroups");
  }
  if (plan.track_barriers && !facts.has_admitted_barrier) {
    plan.track_barriers = false;
    plan.warnings.emplace_back(
        "ConSan MOI skipped barrier tracking for a code object with no admitted barrier sites");
  }
  plan.atomic_or_fence_relevant = plan.track_atomics && facts.has_admitted_atomic;
  if (plan.track_atomics && !plan.atomic_or_fence_relevant) {
    for (const ConSanAtomicSiteDecision &decision : observation_plan.atomic_site_decisions) {
      if (decision.kind != ConSanSiteDecisionKind::Unsupported)
        continue;
      for (const std::string &container_name : decision.source_containers) {
        plan.warnings.emplace_back("ConSan MOI inline atomic ordering skipped " +
                                   std::string(consan_atomic_policy_reason_name(decision.reason)) +
                                   " in " + container_name);
      }
    }
    plan.track_atomics = false;
    plan.warnings.emplace_back(
        "ConSan MOI skipped atomic ordering instrumentation for a code object with no relevant "
        "atomic sites");
  }
  plan.inline_access_present = facts.has_access_candidate;
  plan.inline_atomic_without_access =
      !facts.has_access_candidate && plan.track_atomics && plan.atomic_or_fence_relevant;
  return plan;
}

void apply_inline_shadow_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                                      rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                      std::span<const ConSanMoiCandidate> candidates,
                                      const MoiObjectFacts &, ConSanTransformArtifacts &result) {
  try_apply_inline_shadow_patch(bytes, options, arch, resource_state, candidates, result);
  if (!result.errors.empty())
    return;
  try_apply_inline_shadow_barrier_patch(bytes, options, arch, resource_state, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(bytes, options, arch, result);
}

uint16_t inline_shadow_access_scratch_vgpr_count(const ConSanRequest &request,
                                                 const BoundRuntimeResources &,
                                                 const ConSanMoiOperatingPoint &point,
                                                 const ConSanMoiCandidate &candidate,
                                                 rj_code_arch_t arch) {
  return inline_shadow_scratch_count(request, point, candidate, arch);
}

MoiPersistentStateDemand
plan_inline_shadow_persistent_state_demand(const ConSanRequest &, const BoundRuntimeResources &,
                                           const ConSanMoiOperatingPoint &point,
                                           const MoiPersistentStateFacts &facts) {
  return {
      .needs_workgroup_key = facts.access_count || facts.atomic_count,
      .needs_persistent_state = true,
      .private_dispatch_incompatible_with_dynamic_stack =
          point.automatic_moi_private_dispatch_id && facts.has_operational_dynamic_stack_owner,
      .prefer_private_epoch_for_dynamic_lds = facts.has_operational_dynamic_lds_owner,
      .scalar_state_supported = true,
      .private_state_supported = true,
      .scalar_state_required_for_private_or_overflow = facts.has_operational_dynamic_stack_owner,
  };
}

MoiOperandOverlapSpillPolicy
inline_shadow_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  MoiOperandOverlapSpillPolicy policy;
  policy.supported = context.access_candidate != nullptr &&
                     context.site_kind == ConSanResourceSiteKind::Access &&
                     !context.guest_replay_requires_disjoint_address_scratch;
  return policy;
}

std::optional<uint16_t>
inline_shadow_access_spill_fallback(const MoiAccessSpillFallbackContext &context) {
  if (!consan_uses_gfx9_cdna_encoding(context.arch) || context.candidate.is_flat() ||
      !moi_load_clobbers_address(context.candidate) ||
      candidate_requires_flat_address_materialization(context.candidate) ||
      (!context.spill_required && !context.no_ordinary_window)) {
    return std::nullopt;
  }
  // The compact transaction omits a dedicated address snapshot. Emission
  // either recovers a spilled address or consumes the original address before
  // the deferred application load.
  return inline_shadow_spill_backed_scratch_count(context.request, context.point, context.candidate,
                                                  context.arch);
}

MoiDispatchIdentityPlan
plan_inline_shadow_dispatch_identity(const ConSanRequest &request,
                                     const MoiDispatchIdentityFacts &facts) {
  return {
      .needs_dispatch_id =
          !(facts.target_uses_gfx12_cdna_execution && !request.moi_track_atomics) &&
          facts.has_access_or_atomic_consumer,
      .fallback_kind = std::nullopt,
      .fallback_replans_dispatch_only = false,
      .fallback_diagnostic = {},
  };
}

MoiScalarAbiPlan plan_inline_shadow_scalar_abi(const ConSanRequest &,
                                               const ConSanMoiOperatingPoint &point) {
  std::optional<consan_detail::MoiSpecialStateSgprs> special_state;
  if (point.moi_exec_save_sgpr) {
    const uint16_t base = *point.moi_exec_save_sgpr;
    special_state = consan_detail::MoiSpecialStateSgprs{
        .vcc_save_sgpr = static_cast<uint16_t>(base + 8u),
        .scc_save_sgpr = static_cast<uint16_t>(base + 10u),
    };
  }
  return make_moi_scalar_abi_plan(point, special_state, 12u, true);
}

const MoiModeOperations kInlineShadowModeOperations = {
    plan_inline_shadow_object_mode,
    apply_inline_shadow_mode_patches,
    inline_shadow_access_scratch_vgpr_count,
    plan_inline_shadow_persistent_state_demand,
    true,
    false,
    inline_shadow_operand_overlap_spill,
    inline_shadow_access_spill_fallback,
    plan_inline_shadow_dispatch_identity,
    plan_inline_shadow_scalar_abi,
};

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_atomic.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu
