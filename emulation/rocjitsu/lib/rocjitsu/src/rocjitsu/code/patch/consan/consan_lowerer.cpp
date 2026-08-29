// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This translation unit is the internal lowering façade. Top-level staging is
// a separate compiled composition component, so callers do not gain textual
// visibility into its analysis, mutation, placement, or engine dependencies.

#include "rocjitsu/code/patch/consan/consan_lowerer.h"
#include "rocjitsu/code/patch/consan/consan_composition.h"

namespace rocjitsu {

ConSanTransformArtifacts run_consan_lowering_core(
    std::span<const uint8_t> code_object_bytes, const MoiOptions &options,
    ConSanPerturbationPlanningState *inspected_perturbation,
    const ConSanPreappliedMutationLayout &preapplied_mutation,
    std::span<const ConSanMoiTransientSgprAssignment> initial_owner_transient_sgprs,
    ConSanLoweringExecution *execution, ConSanLoweringExtent extent,
    const ConSanLoweringObservation *observation) {
  return compose_consan_lowering(code_object_bytes, options, inspected_perturbation,
                                 preapplied_mutation, initial_owner_transient_sgprs, execution,
                                 extent, observation);
}

bool install_consan_lowering_observation(const ConSanOptions &options,
                                         ConSanTransformArtifacts &result,
                                         ConSanLoweringExecution *execution,
                                         const ConSanLoweringObservation *prepared_observation) {
  return compose_consan_observation(options, result, execution, prepared_observation);
}

void apply_consan_fault_mutation_plans(std::span<const uint8_t> code_object_bytes,
                                       const ConSanOptions &context,
                                       std::span<const ConSanFaultMutationPlan> plans,
                                       ConSanTransformArtifacts &result) {
  compose_consan_fault_mutation(code_object_bytes, context, plans, result);
}

} // namespace rocjitsu
