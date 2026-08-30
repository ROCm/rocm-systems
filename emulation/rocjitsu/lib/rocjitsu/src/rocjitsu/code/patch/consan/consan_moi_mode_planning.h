// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_mode_planning.h
/// @brief Target-neutral per-object demand selected by one MOI engine.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

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

[[nodiscard]] MoiObjectModePlan plan_record_replay_object_mode(const ConSanRequest &request,
                                                               const ConSanMoiOperatingPoint &point,
                                                               const MoiObjectFacts &facts);

[[nodiscard]] MoiObjectModePlan plan_sampled_object_mode(const ConSanRequest &request,
                                                         const ConSanMoiOperatingPoint &point,
                                                         const MoiObjectFacts &facts);

[[nodiscard]] MoiObjectModePlan
plan_inline_shadow_object_mode(const ConSanRequest &request, const ConSanMoiOperatingPoint &point,
                               const MoiObjectFacts &facts,
                               const ConSanObservationPlan &observation_plan);

} // namespace rocjitsu::consan_moi_impl
