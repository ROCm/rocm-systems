// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowerer.h
/// @brief Internal façade for ConSan mutation and instrumentation lowering.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"

#include <span>

namespace rocjitsu {

[[nodiscard]] ConSanTransformArtifacts run_consan_lowering_core(
    std::span<const uint8_t> code_object_bytes, const MoiOptions &options,
    ConSanPerturbationPlanningState *inspected_perturbation = nullptr,
    const ConSanPreappliedMutationLayout &preapplied_mutation = {},
    std::span<const ConSanMoiTransientSgprAssignment> initial_owner_transient_sgprs = {},
    ConSanLoweringExecution *execution = nullptr,
    ConSanLoweringExtent extent = ConSanLoweringExtent::Complete,
    const ConSanLoweringObservation *observation = nullptr);

[[nodiscard]] bool
install_consan_lowering_observation(const ConSanOptions &options, ConSanTransformArtifacts &result,
                                    ConSanLoweringExecution *execution,
                                    const ConSanLoweringObservation *prepared_observation);

void apply_consan_fault_mutation_plans(std::span<const uint8_t> code_object_bytes,
                                       const ConSanOptions &context,
                                       std::span<const ConSanFaultMutationPlan> plans,
                                       ConSanTransformArtifacts &result);

} // namespace rocjitsu
