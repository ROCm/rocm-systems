// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_composition.h
/// @brief Private contract for coordinating ConSan lowering components.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"

#include <span>

namespace rocjitsu {

/// Coordinate analysis, mutation, observation, and engine lowering through
/// their compiled contracts. This private boundary deliberately exposes only
/// complete top-level products; component-private staging state remains owned
/// by the composition implementation.
[[nodiscard]] ConSanTransformArtifacts
compose_consan_lowering(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                        const ConSanMoiOperatingPoint &initial_operating_point,
                        ConSanPerturbationPlanningState *inspected_perturbation,
                        const ConSanPreappliedMutationLayout &preapplied_mutation,
                        ConSanLoweringExecution *execution, ConSanLoweringExtent extent,
                        const ConSanLoweringObservation *observation);

[[nodiscard]] bool
compose_consan_observation(const ConSanOptions &options, ConSanTransformArtifacts &result,
                           ConSanLoweringExecution *execution,
                           const ConSanLoweringObservation *prepared_observation);

void compose_consan_fault_mutation(std::span<const uint8_t> code_object_bytes,
                                   const ConSanOptions &context,
                                   std::span<const ConSanFaultMutationPlan> plans,
                                   ConSanTransformArtifacts &result);

} // namespace rocjitsu
