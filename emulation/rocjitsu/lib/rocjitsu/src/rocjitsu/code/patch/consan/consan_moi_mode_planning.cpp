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
  plan.dense_barrier_router = point.moi_record_replay_dense_barrier_router;
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

MoiObjectModePlan plan_moi_object_mode(const ConSanRequest &request,
                                       const ConSanMoiOperatingPoint &point,
                                       const MoiObjectFacts &facts,
                                       const ConSanObservationPlan &observation_plan) {
  return moi_mode_operations(request.moi_engine).plan(request, point, facts, observation_plan);
}

void apply_moi_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                            rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, ConSanTransformArtifacts &result) {
  moi_mode_operations(options.moi_engine)
      .apply(bytes, options, arch, resource_state, candidates, facts, result);
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
make_moi_scalar_abi_plan(const ConSanMoiOperatingPoint &point,
                         std::optional<consan_detail::MoiSpecialStateSgprs> special_state,
                         uint16_t fixed_indirect_pc_offset, bool access_router_uses_dense_abi) {
  MoiScalarAbiPlan plan{.special_state = special_state,
                        .indirect_jump = std::nullopt,
                        .access_router_uses_dense_abi = access_router_uses_dense_abi};
  if (point.has_moi_scalar_spill()) {
    if (point.moi_router_indirect_pc_sgpr && point.moi_router_indirect_scc_sgpr) {
      plan.indirect_jump =
          MoiIndirectJumpSgprs{.pc_sgpr = *point.moi_router_indirect_pc_sgpr,
                               .scc_save_sgpr = *point.moi_router_indirect_scc_sgpr};
    }
    return plan;
  }
  if (!point.moi_exec_save_sgpr || !special_state)
    return plan;
  plan.indirect_jump = MoiIndirectJumpSgprs{
      .pc_sgpr = static_cast<uint16_t>(*point.moi_exec_save_sgpr + fixed_indirect_pc_offset),
      .scc_save_sgpr = special_state->scc_save_sgpr,
  };
  return plan;
}

MoiScalarAbiPlan plan_moi_scalar_abi(const ConSanRequest &request,
                                     const ConSanMoiOperatingPoint &point) {
  return moi_mode_operations(request.moi_engine).scalar_abi(request, point);
}

ConSanEvidenceRequirements
plan_moi_evidence_requirements(ConSanMoiEngine engine, const MoiEvidencePlanningContext &context) {
  return moi_mode_operations(engine).plan_evidence(context);
}

std::optional<consan_detail::MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanRequest &request, const ConSanMoiOperatingPoint &point) {
  return plan_moi_scalar_abi(request, point).special_state;
}

std::optional<MoiIndirectJumpSgprs> moi_indirect_jump_sgprs(const ConSanRequest &request,
                                                            const ConSanMoiOperatingPoint &point) {
  return plan_moi_scalar_abi(request, point).indirect_jump;
}

} // namespace rocjitsu::consan_moi_impl
