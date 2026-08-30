// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"

namespace rocjitsu::consan_moi_impl {

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

MoiObjectModePlan plan_moi_object_mode(const ConSanRequest &request,
                                       const ConSanMoiOperatingPoint &point,
                                       const MoiObjectFacts &facts,
                                       const ConSanObservationPlan &observation_plan) {
  switch (request.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    return plan_record_replay_object_mode(request, point, facts);
  case ConSanMoiEngine::Sampled:
    return plan_sampled_object_mode(request, point, facts);
  case ConSanMoiEngine::InlineShadow:
    return plan_inline_shadow_object_mode(request, point, facts, observation_plan);
  }
}

void apply_moi_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                            rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, ConSanTransformArtifacts &result) {
  switch (options.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    apply_record_replay_mode_patches(bytes, options, arch, resource_state, candidates, facts,
                                     result);
    return;
  case ConSanMoiEngine::Sampled:
    apply_sampled_mode_patches(bytes, options, arch, resource_state, candidates, result);
    return;
  case ConSanMoiEngine::InlineShadow:
    apply_inline_shadow_mode_patches(bytes, options, arch, resource_state, candidates, result);
    return;
  }
}

} // namespace rocjitsu::consan_moi_impl
