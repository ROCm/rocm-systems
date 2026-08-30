// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_mode_planning.h
/// @brief Target-neutral per-object demand selected by one MOI engine.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

#include <cstddef>
#include <string>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Shared inventory facts presented to every MOI engine. Target inspection is
/// normalized before this boundary; mode owners never decode an ISA family.
struct MoiObjectFacts {
  bool has_access_candidate = false;
  bool has_admitted_atomic = false;
  bool has_admitted_fence = false;
  bool has_admitted_barrier = false;
  size_t admitted_barrier_count = 0;
  bool has_stranded_admitted_barrier = false;
  bool target_supports_dense_barrier_router = false;
  bool has_explicit_persistent_state = false;
};

/// Effective mode demand consumed by common resource solving. This deliberately
/// contains only the request/operating-point fields that an engine may refine
/// after seeing one object's semantic inventory.
struct MoiObjectModePlan {
  ConSanMoiOwnerSource owner_source = ConSanMoiOwnerSource::Automatic;
  bool track_atomics = false;
  bool track_barriers = false;
  std::optional<bool> initialize_owner_epoch;
  bool dense_barrier_router = false;
  bool inline_access_present = false;
  bool inline_atomic_without_access = false;
  bool atomic_or_fence_relevant = false;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

[[nodiscard]] MoiObjectModePlan
make_moi_object_mode_plan(const ConSanRequest &request, const ConSanMoiOperatingPoint &point,
                          ConSanMoiOwnerSource automatic_owner_source);

/// The only cross-engine selection point for per-object demand. Adding an MOI
/// engine requires one engine-owned planner and one entry in this dispatcher;
/// common resource solving consumes the same narrow product unchanged.
[[nodiscard]] MoiObjectModePlan plan_moi_object_mode(const ConSanRequest &request,
                                                     const ConSanMoiOperatingPoint &point,
                                                     const MoiObjectFacts &facts,
                                                     const ConSanObservationPlan &observation_plan);

/// Run the selected engine's lowering sequence. Shared placement has already
/// accepted an operating point; the engine owns which access and sync
/// consumers run, their order, and any mode-local post-placement cleanup.
void apply_moi_mode_patches(std::span<const uint8_t> bytes, MoiOptions &options,
                            rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                            std::span<const ConSanMoiCandidate> candidates,
                            const MoiObjectFacts &facts, ConSanTransformArtifacts &result);

struct MoiModeOperations {
  MoiObjectModePlan (*plan)(const ConSanRequest &, const ConSanMoiOperatingPoint &,
                            const MoiObjectFacts &, const ConSanObservationPlan &);
  void (*apply)(std::span<const uint8_t>, MoiOptions &, rj_code_arch_t, MoiResourcePlanningState &,
                std::span<const ConSanMoiCandidate>, const MoiObjectFacts &,
                ConSanTransformArtifacts &);
  uint16_t (*access_scratch_vgpr_count)(const ConSanRequest &, const BoundRuntimeResources &,
                                        const ConSanMoiOperatingPoint &, const ConSanMoiCandidate &,
                                        rj_code_arch_t);
};

extern const MoiModeOperations kRecordReplayModeOperations;
extern const MoiModeOperations kSampledModeOperations;
extern const MoiModeOperations kInlineShadowModeOperations;

[[nodiscard]] const MoiModeOperations &moi_mode_operations(ConSanMoiEngine engine);

} // namespace rocjitsu::consan_moi_impl
