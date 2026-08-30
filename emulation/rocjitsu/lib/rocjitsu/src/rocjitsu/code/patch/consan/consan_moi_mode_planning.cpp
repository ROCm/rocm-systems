// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"

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
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return kRecordReplayModeOperations;
  case ConSanMoiEngine::Sampled:
    return kSampledModeOperations;
  case ConSanMoiEngine::InlineShadow:
    return kInlineShadowModeOperations;
  }
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
  const MoiDynamicStackSpillFacts facts{
      .target_has_backend = consan_is_capability_arch(arch),
  };
  return moi_mode_operations(engine).dynamic_stack_spill(facts);
}

MoiOperandOverlapSpillPolicy
plan_moi_operand_overlap_spill(const MoiOperandOverlapSpillContext &context) {
  return moi_mode_operations(context.request.moi_engine).operand_overlap_spill(context);
}

} // namespace rocjitsu::consan_moi_impl
