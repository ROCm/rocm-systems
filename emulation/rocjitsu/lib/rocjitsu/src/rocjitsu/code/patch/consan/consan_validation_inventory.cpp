// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis.h"

#include <memory>
#include <utility>

namespace rocjitsu::consan {

namespace {

[[nodiscard]] ProgramAnalysisResult rederive_pristine_program_inventory(
    std::span<const uint8_t> original_image, const Request &request, const DebugOverrides &debug,
    const MutationRequest &mutation,
    SuperColliderPerturbationPlanningState &supercollider_perturbation) {
  ProgramAnalysisResult analysis;
  ProgramInventoryBuilder inventory_builder(original_image);
  std::unique_ptr<AmdGpuCodeObject> code_object;
  (void)analyze_program_inventory(original_image, request, debug, mutation, code_object,
                                  inventory_builder, supercollider_perturbation, analysis);
  return analysis;
}

} // namespace

PristineValidationInventory
rederive_pristine_validation_inventory(std::span<const uint8_t> original_image,
                                       bool require_mutation_semantics) {
  Request request;
  request.mode = Mode::SuperCollider;
  DebugOverrides debug;
  MutationRequest mutation;
  mutation.fault_drop_barrier = require_mutation_semantics;
  mutation.fault_dry_run = true;
  SuperColliderPerturbationPlanningState supercollider_perturbation;
  ProgramAnalysisResult analysis = rederive_pristine_program_inventory(
      original_image, request, debug, mutation, supercollider_perturbation);
  return {
      .program_inventory = std::move(analysis.program_inventory),
      .fault_sites = std::move(analysis.fault_sites),
      .supercollider_perturbation_candidates = std::move(supercollider_perturbation.candidates),
      .analysis_succeeded = analysis.errors.empty(),
  };
}

} // namespace rocjitsu::consan
