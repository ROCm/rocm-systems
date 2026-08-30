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

} // namespace rocjitsu::consan_moi_impl
