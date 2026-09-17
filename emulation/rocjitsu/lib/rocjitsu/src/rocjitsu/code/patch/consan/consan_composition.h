// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_composition.h
/// @brief Private contract for coordinating ConSan lowering components.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"

#include <span>

namespace rocjitsu::consan {

/// Coordinate analysis, mutation, observation, and mode lowering through
/// their compiled contracts. This private boundary deliberately exposes only
/// complete top-level products; component-private staging state remains owned
/// by the composition implementation.
[[nodiscard]] TransformArtifacts
compose_lowering(std::span<const uint8_t> code_object_bytes, const Options &options,
                 const OperatingPoint &initial_operating_point,
                 SuperColliderPerturbationPlanningState *inspected_supercollider_perturbation,
                 const PreappliedMutationLayout &preapplied_mutation, LoweringExtent extent,
                 const LoweringObservation *observation);

[[nodiscard]] bool compose_observation(const Options &options, TransformArtifacts &result,
                                       const LoweringObservation *prepared_observation);

} // namespace rocjitsu::consan
