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
#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"
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

MoiInlineShadowScalarState
project_inline_shadow_scalar_state(const ConSanMoiOperatingPoint &point) {
  return {
      .exec_save_sgpr = point.moi_exec_save_sgpr,
      .router_call = point.moi_router_call,
      .visible_evidence_sgpr = point.moi_inline_visible_evidence_sgpr,
      .inline_scalar_spill = point.has_inline_moi_scalar_spill(),
      .scalar_spill = point.has_moi_scalar_spill(),
      .branch_only_spill = point.moi_branch_only_spill.has_value(),
      .exec_save_persistent = point.moi_exec_save_sgprs_persistent,
      .dynamic_stack_spill = point.moi_dynamic_stack_spill,
  };
}

std::optional<uint16_t>
inline_shadow_visible_evidence_sgpr(const MoiInlineShadowScalarState &state) {
  if (!state.exec_save_sgpr)
    return std::nullopt;
  if (state.inline_scalar_spill)
    return state.visible_evidence_sgpr;
  if (!state.exec_save_persistent)
    return std::nullopt;
  return static_cast<uint16_t>(*state.exec_save_sgpr + (state.dynamic_stack_spill ? 25u : 24u));
}

uint16_t inline_shadow_loop_scratch_count(const ConSanMoiCandidate &candidate) {
  // Wide local accesses retain an offset and iteration counter. Wide external
  // accesses retain an iteration counter and the workgroup key, since the
  // versioned transaction reuses all of its ordinary address temporaries.
  return consan_detail::inline_shadow_loop_scratch_count(candidate.width_bits(),
                                                         consan_moi_shadow_cell::granule_bytes);
}

bool validate_inline_shadow_exec_save_sgpr(const MoiInlineShadowScalarState &state,
                                           bool inline_access_present, uint16_t required_sgpr_count,
                                           rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!state.exec_save_sgpr)
    return true;
  if (state.inline_scalar_spill && inline_access_present &&
      (!state.router_call && !state.branch_only_spill)) {
    errors.emplace_back(
        "ConSan MOI spill-backed inline-shadow probes require a dense router or branch-only "
        "scalar spill");
    return false;
  }
  const uint16_t ordinary_sgpr_count = consan_uses_gfx9_cdna_encoding(arch) ? 102u : kMaxSgprs;
  const uint16_t max_exec_save_sgpr =
      static_cast<uint16_t>(ordinary_sgpr_count - required_sgpr_count);
  if (*state.exec_save_sgpr > max_exec_save_sgpr || *state.exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline-shadow diagnostics require an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0.." +
                        std::to_string(max_exec_save_sgpr));
    return false;
  }
  return true;
}

uint16_t inline_shadow_scratch_count(bool track_atomics,
                                     const MoiAccessResourceFacts &resource_facts,
                                     const ConSanMoiCandidate &candidate) {
  return static_cast<uint16_t>(consan_detail::inline_shadow_transaction_scratch_count(
                                   resource_facts.has_exec_save, track_atomics) +
                               inline_shadow_loop_scratch_count(candidate) +
                               resource_facts.address_scratch_vgpr_count +
                               resource_facts.dynamic_stack_reservoir_vgpr_count +
                               resource_facts.two_address_replay_vgpr_count);
}

uint16_t inline_shadow_spill_backed_scratch_count(bool track_atomics,
                                                  const MoiAccessResourceFacts &resource_facts,
                                                  const ConSanMoiCandidate &candidate) {
  const uint16_t normal_count =
      inline_shadow_scratch_count(track_atomics, resource_facts, candidate);
  // A spill-backed CDNA probe can recover an overlapping guest address from
  // its authoritative private slot into a phase-shared transaction register.
  return static_cast<uint16_t>(normal_count -
                               (resource_facts.supports_clobbered_address_spill_reload ? 1u : 0u));
}

MoiObjectModePlan plan_inline_shadow_object_mode(const ConSanRequest &request,
                                                 const BoundRuntimeResources &resources,
                                                 const TransformPolicy &,
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
  if (plan.track_atomics && !facts.has_admitted_atomic) {
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
  plan.semantics.inline_access_present = facts.has_access_candidate;
  plan.semantics.report_layout = resolve_moi_report_layout(
      resources, ConSanMoiEngine::InlineShadow,
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(resources.moi_report_buffer_size));
  plan.inline_atomic_without_access =
      !facts.has_access_candidate && plan.track_atomics && facts.has_admitted_atomic;
  return plan;
}

void apply_inline_shadow_mode_patches(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                      ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
                                      MoiResourcePlanningState &resource_state,
                                      std::span<const ConSanMoiCandidate> candidates,
                                      const MoiObjectFacts &,
                                      const MoiObjectModeSemantics &semantics,
                                      ConSanTransformArtifacts &result) {
  try_apply_inline_shadow_patch(bytes, options, operating_point, arch, resource_state, candidates,
                                semantics, result);
  if (!result.errors.empty())
    return;
  try_apply_inline_shadow_barrier_patch(bytes, options, operating_point, arch, resource_state,
                                        semantics, result);
  if (result.errors.empty())
    try_apply_inline_atomic_ordering_patch(bytes, options, operating_point, arch, semantics,
                                           result);
}

uint16_t inline_shadow_access_scratch_vgpr_count(const ConSanRequest &request,
                                                 const BoundRuntimeResources &,
                                                 const MoiAccessResourceFacts &resource_facts,
                                                 const ConSanMoiCandidate &candidate) {
  return inline_shadow_scratch_count(request.moi_track_atomics, resource_facts, candidate);
}

MoiPersistentStateDemand
plan_inline_shadow_persistent_state_demand(const ConSanRequest &, const BoundRuntimeResources &,
                                           const ConSanMoiOperatingPoint &point,
                                           const MoiPersistentStateFacts &facts) {
  return {
      .needs_workgroup_key = facts.access_count || facts.atomic_count,
      .needs_persistent_state = true,
      .private_dispatch_incompatible_with_dynamic_stack =
          point.moi_dispatch_identity.private_fallback() &&
          facts.has_operational_dynamic_stack_owner,
      .prefer_private_epoch_for_dynamic_lds = facts.has_operational_dynamic_lds_owner,
      .scalar_state_supported = true,
      .private_state_supported = true,
      .scalar_state_required_for_private_or_overflow = facts.has_operational_dynamic_stack_owner,
      .cdna_overflow_strategy = MoiCdnaPersistentOverflowStrategy::ResidentWavePrivateState,
  };
}

MoiOperandOverlapSpillPolicy
inline_shadow_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  MoiOperandOverlapSpillPolicy policy;
  policy.supported = context.access_candidate != nullptr &&
                     context.site_kind == ConSanResourceSiteKind::Access &&
                     !context.resource_facts.guest_replay_requires_disjoint_address_scratch;
  return policy;
}

std::optional<uint16_t>
inline_shadow_access_spill_fallback(const MoiAccessSpillFallbackContext &context) {
  if (!context.resource_facts.supports_clobbered_address_spill_reload ||
      (!context.spill_required && !context.no_ordinary_window)) {
    return std::nullopt;
  }
  // The compact transaction omits a dedicated address snapshot. Emission
  // either recovers a spilled address or consumes the original address before
  // the deferred application load.
  return inline_shadow_spill_backed_scratch_count(context.request.moi_track_atomics,
                                                  context.resource_facts, context.candidate);
}

MoiDispatchIdentityPlan
plan_inline_shadow_dispatch_identity(const ConSanRequest &request,
                                     const MoiDispatchIdentityFacts &facts) {
  MoiDispatchIdentityPlan plan;
  plan.needs_dispatch_id =
      (facts.access_reports_need_explicit_identity || request.moi_track_atomics) &&
      facts.has_access_or_atomic_consumer;
  plan.permits_private_entry_capture = true;
  return plan;
}

MoiScalarAbiPlan plan_inline_shadow_scalar_abi(const MoiScalarRoutingState &routing_state) {
  std::optional<consan_detail::MoiSpecialStateSgprs> special_state;
  if (routing_state.exec_save_sgpr) {
    const uint16_t base = *routing_state.exec_save_sgpr;
    special_state = consan_detail::MoiSpecialStateSgprs{
        .vcc_save_sgpr = static_cast<uint16_t>(base + 8u),
        .scc_save_sgpr = static_cast<uint16_t>(base + 10u),
    };
  }
  return make_moi_scalar_abi_plan(routing_state, special_state, 12u, true);
}

std::optional<MoiDenseRouterPlan>
plan_inline_shadow_dense_router(const MoiScalarAbiPlan &scalar_abi,
                                const MoiScalarRoutingState &routing_state,
                                const MoiScalarTargetFacts &target) {
  if (routing_state.has_branch_only_spill)
    return std::nullopt;
  if (!scalar_abi.indirect_jump)
    return std::nullopt;

  uint16_t dispatch_key_sgpr = 0u;
  std::optional<uint16_t> call_return_sgpr;
  if (routing_state.has_inline_spill()) {
    if (!routing_state.router_call)
      return std::nullopt;
    dispatch_key_sgpr = routing_state.router_call->dispatch_key_sgpr;
    if (target.direct_call_form == ConSanDirectCallForm::SCallI64)
      call_return_sgpr = routing_state.router_call->call_return_sgpr;
  } else {
    if (!routing_state.exec_save_sgpr)
      return std::nullopt;
    const uint16_t base = *routing_state.exec_save_sgpr;
    dispatch_key_sgpr = target.direct_call_form == ConSanDirectCallForm::SCallI64
                            ? scalar_abi.indirect_jump->pc_sgpr
                            : static_cast<uint16_t>(base + 28u);
    if (target.direct_call_form == ConSanDirectCallForm::SCallI64)
      call_return_sgpr = static_cast<uint16_t>(base + 28u);
  }

  return MoiDenseRouterPlan{
      .indirect_jump = *scalar_abi.indirect_jump,
      .dispatch_key_sgpr = dispatch_key_sgpr,
      .call_return_sgpr = call_return_sgpr,
      .entry_island_words = kMoiInlineShadowIndirectIslandWords,
      .relocated_entry_return_words = kMoiInlineShadowIndirectIslandWords,
      .explicit_key = !call_return_sgpr.has_value(),
      .restore_scc_before_route = true,
      .requires_indirect_pc_wait = true,
      .publish_entry_island_offset = true,
  };
}

uint16_t inline_shadow_exec_save_sgpr_count(const MoiExecSaveRequirement &requirement,
                                            const MoiScalarTargetFacts &) {
  if (!requirement.has_report_buffer)
    return 0u;
  const uint16_t engine_count =
      requirement.inline_access_present ? kConSanMoiInlineExecSaveSgprCount : 22u;
  const uint16_t stack_count = requirement.dynamic_stack_spill ? 25u : 0u;
  return std::max(engine_count, stack_count);
}

const MoiModeOperations kInlineShadowModeOperations = {
    .plan = plan_inline_shadow_object_mode,
    .apply = apply_inline_shadow_mode_patches,
    .access_scratch_vgpr_count = inline_shadow_access_scratch_vgpr_count,
    .operational_evidence = {ConSanProbeIntentKind::ExactBarrierEpoch,
                             ConSanProbeIntentKind::ExactAtomicOrdering, false},
    .dynamic_stack_frame_save_sgpr_offset = 24u,
    .exec_save_sgpr_count = inline_shadow_exec_save_sgpr_count,
    .prologue =
        {
            .skip_unobserved_barrier_only_initialization = true,
            .one_based_owner_ids = true,
            .persistent_state_requires_in_place_entry = true,
        },
    .persistent_state_demand = plan_inline_shadow_persistent_state_demand,
    .transient_scalar_placement = {ConSanMoiScalarSpillLayout::Inline, false, false, 0u},
    .dynamic_stack_spill_without_target_backend = true,
    .dynamic_stack_spill_requires_every_owner_dynamic = false,
    .operand_overlap_spill = inline_shadow_operand_overlap_spill,
    .access_spill_fallback = inline_shadow_access_spill_fallback,
    .dispatch_identity = plan_inline_shadow_dispatch_identity,
    .scalar_abi = plan_inline_shadow_scalar_abi,
    .dense_access_route = {},
    .dense_router = plan_inline_shadow_dense_router,
    .plan_evidence = plan_inline_shadow_evidence_requirements,
    .plan_report_layout = plan_inline_shadow_report_layout,
    .reconstruct_report_inventory = reconstruct_inline_shadow_report_inventory,
};

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_atomic.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu
