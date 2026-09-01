// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis.h"

#include <memory>
#include <utility>

namespace rocjitsu {

namespace {

[[nodiscard]] ConSanProgramAnalysisResult
rederive_pristine_program_inventory(std::span<const uint8_t> original_image,
                                    const ConSanOptions &options,
                                    ConSanPerturbationPlanningState &perturbation) {
  ConSanProgramAnalysisResult analysis;
  ProgramInventoryBuilder inventory_builder(original_image);
  std::unique_ptr<AmdGpuCodeObject> code_object;
  (void)analyze_consan_program_inventory(original_image, options, code_object, inventory_builder,
                                         perturbation, analysis);
  return analysis;
}

} // namespace

ConSanMutationValidationInventory
rederive_consan_mutation_validation_inventory(std::span<const uint8_t> original_image) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  options.fault_dry_run = true;
  ConSanPerturbationPlanningState unused_perturbation;
  ConSanProgramAnalysisResult analysis =
      rederive_pristine_program_inventory(original_image, options, unused_perturbation);
  return {
      .program_inventory = std::move(analysis.program_inventory),
      .fault_sites = std::move(analysis.fault_sites),
  };
}

ConSanPerturbationValidationInventory
rederive_consan_perturbation_validation_inventory(std::span<const uint8_t> original_image) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  ConSanPerturbationPlanningState perturbation;
  ConSanProgramAnalysisResult analysis =
      rederive_pristine_program_inventory(original_image, options, perturbation);
  return {
      .program_inventory = std::move(analysis.program_inventory),
      .candidates = std::move(perturbation.candidates),
      .analysis_succeeded = analysis.errors.empty(),
  };
}

} // namespace rocjitsu
