// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"

#include "rocjitsu/code/patch/consan/consan_composition.h"

#include <utility>

namespace rocjitsu {

ConSanMutationValidationInventory
rederive_consan_mutation_validation_inventory(std::span<const uint8_t> original_image) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  options.fault_dry_run = true;
  ConSanTransformArtifacts artifacts =
      compose_consan_lowering(original_image, options, nullptr, {}, {}, nullptr,
                              ConSanLoweringExtent::ThroughProgramInventory, nullptr);
  return {
      .program_inventory = std::move(artifacts.program_inventory),
      .fault_sites = std::move(artifacts.fault_sites),
  };
}

ConSanPerturbationValidationInventory
rederive_consan_perturbation_validation_inventory(std::span<const uint8_t> original_image) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  ConSanPerturbationPlanningState perturbation;
  ConSanTransformArtifacts artifacts =
      compose_consan_lowering(original_image, options, &perturbation, {}, {}, nullptr,
                              ConSanLoweringExtent::ThroughProgramInventory, nullptr);
  return {
      .program_inventory = std::move(artifacts.program_inventory),
      .candidates = std::move(perturbation.candidates),
      .analysis_succeeded = artifacts.errors.empty(),
  };
}

} // namespace rocjitsu
