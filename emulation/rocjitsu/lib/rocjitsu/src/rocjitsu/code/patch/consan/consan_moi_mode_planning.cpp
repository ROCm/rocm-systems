// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"

#include <array>

namespace rocjitsu::consan_moi_impl {

MoiPersistentStateDemand make_exact_workgroup_capture_demand(const ConSanRequest &request,
                                                             const BoundRuntimeResources &resources,
                                                             const ConSanMoiOperatingPoint &point,
                                                             const MoiPersistentStateFacts &facts) {
  MoiPersistentStateDemand demand;
  demand.needs_entry_workgroup_tuple =
      (facts.access_count || facts.atomic_count || facts.barrier_count || facts.fence_count) &&
      !consan_moi_detail::record_replay_has_entry_workgroup_capture(point);
  demand.private_workgroup_tuple_supported =
      demand.needs_entry_workgroup_tuple &&
      !consan_moi_detail::record_replay_uses_automatic_banked_capture(request, resources);
  return demand;
}

MoiObjectModePlan make_moi_object_mode_plan(const ConSanRequest &request,
                                            const ConSanMoiOperatingPoint &point,
                                            ConSanMoiOwnerSource automatic_owner_source) {
  MoiObjectModePlan plan;
  plan.owner_source = request.moi_owner_source == ConSanMoiOwnerSource::Automatic
                          ? automatic_owner_source
                          : request.moi_owner_source;
  plan.track_atomics = request.moi_track_atomics;
  plan.track_barriers = request.moi_track_barriers;
  plan.initialize_owner_epoch = point.moi_initialize_owner_epoch;
  return plan;
}

const MoiModeOperations &moi_mode_operations(ConSanMoiEngine engine) {
  static constexpr std::array registrations{
      MoiModeRegistration{ConSanMoiEngine::RecordReplay, &kRecordReplayModeOperations},
      MoiModeRegistration{ConSanMoiEngine::Sampled, &kSampledModeOperations},
      MoiModeRegistration{ConSanMoiEngine::InlineShadow, &kInlineShadowModeOperations},
  };
  return *find_moi_mode_operations<ConSanMoiEngine>(registrations, engine);
}

MoiObjectModePlan
plan_moi_object_mode(const ConSanRequest &request, const BoundRuntimeResources &resources,
                     const TransformPolicy &policy, const ConSanMoiOperatingPoint &point,
                     const MoiObjectFacts &facts, const ConSanObservationPlan &observation_plan) {
  return moi_mode_operations(request.moi_engine)
      .plan(request, resources, policy, point, facts, observation_plan);
}

void apply_moi_mode_patches(std::span<const uint8_t> bytes, const ConSanOptions &options,
                            ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
                            MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, const MoiObjectModeSemantics &semantics,
                            ConSanTransformArtifacts &result) {
  moi_mode_operations(options.moi_engine)
      .apply(bytes, options, operating_point, arch, resource_state, candidates, facts, semantics,
             result);
}

MoiPersistentStateDemand plan_moi_persistent_state_demand(const ConSanRequest &request,
                                                          const BoundRuntimeResources &resources,
                                                          const ConSanMoiOperatingPoint &point,
                                                          const MoiPersistentStateFacts &facts) {
  return moi_mode_operations(request.moi_engine)
      .persistent_state_demand(request, resources, point, facts);
}

MoiDynamicStackSpillPolicy plan_moi_dynamic_stack_spill(ConSanMoiEngine engine,
                                                        rj_code_arch_t arch) {
  const MoiModeOperations &operations = moi_mode_operations(engine);
  return {
      .backend_supported =
          operations.dynamic_stack_spill_without_target_backend || consan_is_capability_arch(arch),
      .requires_every_owner_dynamic = operations.dynamic_stack_spill_requires_every_owner_dynamic,
  };
}

MoiOperandOverlapSpillPolicy
plan_moi_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  return moi_mode_operations(context.request.moi_engine).operand_overlap_spill(context);
}

MoiDispatchIdentityPlan plan_moi_dispatch_identity(const ConSanRequest &request,
                                                   const MoiDispatchIdentityFacts &facts) {
  return moi_mode_operations(request.moi_engine).dispatch_identity(request, facts);
}

MoiScalarAbiPlan
make_moi_scalar_abi_plan(const MoiScalarRoutingState &routing_state,
                         std::optional<consan_detail::MoiSpecialStateSgprs> special_state,
                         uint16_t fixed_indirect_pc_offset, bool access_router_uses_dense_abi) {
  MoiScalarAbiPlan plan{.special_state = special_state,
                        .indirect_jump = std::nullopt,
                        .access_router_uses_dense_abi = access_router_uses_dense_abi};
  if (routing_state.has_scalar_spill()) {
    if (routing_state.router_jump)
      plan.indirect_jump = routing_state.router_jump;
    return plan;
  }
  if (!routing_state.exec_save_sgpr || !special_state)
    return plan;
  plan.indirect_jump = ConSanMoiIndirectJumpSgprs{
      .pc_sgpr = static_cast<uint16_t>(*routing_state.exec_save_sgpr + fixed_indirect_pc_offset),
      .scc_save_sgpr = special_state->scc_save_sgpr,
  };
  return plan;
}

MoiScalarAbiPlan plan_moi_scalar_abi(const ConSanRequest &request,
                                     const ConSanMoiOperatingPoint &point) {
  return moi_mode_operations(request.moi_engine)
      .scalar_abi(project_moi_scalar_routing_state(point));
}

std::optional<MoiDenseRouterPlan>
make_recording_moi_dense_router_plan(const MoiScalarAbiPlan &scalar_abi,
                                     const MoiScalarRoutingState &routing_state,
                                     const MoiScalarTargetFacts &target) {
  if (!routing_state.exec_save_sgpr || !scalar_abi.special_state || !scalar_abi.indirect_jump ||
      routing_state.has_branch_only_spill) {
    return std::nullopt;
  }
  const bool spill_backed = routing_state.has_compact_spill();
  if (spill_backed && (!routing_state.router_jump || !routing_state.router_call))
    return std::nullopt;

  const uint16_t base = *routing_state.exec_save_sgpr;
  const uint16_t dispatch_key_sgpr = spill_backed ? routing_state.router_call->dispatch_key_sgpr
                                                  : static_cast<uint16_t>(base + 5u);
  const uint16_t call_return_sgpr =
      spill_backed ? routing_state.router_call->call_return_sgpr : static_cast<uint16_t>(base + 6u);
  const bool explicit_key = target.direct_call_form != ConSanDirectCallForm::SCallI64;
  return MoiDenseRouterPlan{
      .indirect_jump = *scalar_abi.indirect_jump,
      .dispatch_key_sgpr = dispatch_key_sgpr,
      .call_return_sgpr = call_return_sgpr,
      .entry_island_words = moi_record_replay_entry_island_words(spill_backed),
      .relocated_entry_return_words = kMoiRecordReplayIndirectIslandWords,
      .explicit_key = explicit_key,
      .collapse_spill_router =
          spill_backed && call_return_sgpr == scalar_abi.indirect_jump->pc_sgpr && !explicit_key,
      .restore_scc_before_route = explicit_key,
  };
}

std::optional<MoiDenseRouterPlan> plan_moi_dense_router(const ConSanRequest &request,
                                                        const ConSanMoiOperatingPoint &point,
                                                        rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  const MoiModeOperations &operations = moi_mode_operations(request.moi_engine);
  if (target == nullptr || operations.dense_router == nullptr ||
      (operations.dense_access_route.requires_target_dense_call_capability &&
       target->direct_call_form != ConSanDirectCallForm::SCallI64 &&
       !target->supports_moi_dense_s_call_b64)) {
    return std::nullopt;
  }
  const MoiScalarRoutingState routing_state = project_moi_scalar_routing_state(point);
  const MoiScalarTargetFacts target_facts{.direct_call_form = target->direct_call_form};
  return operations.dense_router(operations.scalar_abi(routing_state), routing_state, target_facts);
}

MoiDenseAccessRouteTraits moi_dense_access_route_traits(ConSanMoiEngine engine) {
  return moi_mode_operations(engine).dense_access_route;
}

ConSanEvidenceRequirements
plan_moi_evidence_requirements(ConSanMoiEngine engine, const MoiEvidencePlanningContext &context) {
  return moi_mode_operations(engine).plan_evidence(context);
}

std::optional<consan_detail::MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanRequest &request, const ConSanMoiOperatingPoint &point) {
  return plan_moi_scalar_abi(request, point).special_state;
}

std::optional<ConSanMoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const ConSanRequest &request, const ConSanMoiOperatingPoint &point) {
  return plan_moi_scalar_abi(request, point).indirect_jump;
}

} // namespace rocjitsu::consan_moi_impl
