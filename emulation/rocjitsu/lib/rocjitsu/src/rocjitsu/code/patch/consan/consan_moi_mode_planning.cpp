// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"

#include <array>

namespace rocjitsu::consan_moi_impl {

MoiTargetFacts resolve_moi_target_facts(rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target ? MoiTargetFacts{true, target->direct_call_form,
                                 target->flat_compare_swap_data_pair_alignment > 1u}
                : MoiTargetFacts{};
}

MoiPersistentStateDemand make_exact_workgroup_capture_demand(const ConSanMoiOperatingPoint &point,
                                                             const MoiPersistentStateFacts &facts) {
  MoiPersistentStateDemand demand;
  demand.needs_entry_workgroup_tuple =
      (facts.access_count || facts.atomic_count || facts.barrier_count || facts.fence_count) &&
      !consan_moi_detail::moi_has_exact_entry_workgroup_capture(point);
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
  const MoiModeOperations &operations = moi_mode_operations(request.moi_engine);
  MoiObjectModePlan plan =
      operations.plan(request, resources, policy, point, facts, observation_plan);
  plan.prologue = operations.prologue;
  return plan;
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

MoiScalarAbiPlan plan_moi_scalar_abi(ConSanMoiEngine engine,
                                     const MoiScalarPreservationState &preservation_state) {
  return moi_mode_operations(engine).scalar_abi(preservation_state);
}

ConSanEvidenceRequirements
plan_moi_evidence_requirements(ConSanMoiEngine engine, const MoiEvidencePlanningContext &context) {
  return moi_mode_operations(engine).plan_evidence(context);
}

} // namespace rocjitsu::consan_moi_impl

rocjitsu::ConSanMoiModePolicy rocjitsu::consan_moi_mode_policy(ConSanMoiEngine engine) {
  return consan_moi_impl::moi_mode_operations(engine).policy;
}
