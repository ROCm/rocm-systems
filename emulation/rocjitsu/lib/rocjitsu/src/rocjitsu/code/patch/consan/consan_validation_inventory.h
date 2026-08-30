// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_inventory.h
/// @brief Narrow pristine-analysis inputs rederived for final validation.

#pragma once

#include "rocjitsu/code/patch/consan/consan_fault_selection.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

/// Exact immutable facts rederived from the pristine code object for mutation
/// proof. Mutation analysis deliberately enables extended barrier pairing.
struct ConSanMutationValidationInventory {
  ProgramInventory program_inventory;
  std::vector<ConSanFaultSite> fault_sites;

  [[nodiscard]] ConSanFaultSelectionView fault_selection() const {
    return {.program_inventory = program_inventory, .fault_sites = fault_sites};
  }
};

/// Rebuild the pristine barrier and fault inventory needed to prove an exact
/// mutation. No planning or mutation bytes are applied.
[[nodiscard]] ConSanMutationValidationInventory
rederive_consan_mutation_validation_inventory(std::span<const uint8_t> original_image);

/// Exact ordinary synchronization inventory and perturbation candidates
/// rederived independently from construction.
struct ConSanPerturbationValidationInventory {
  ProgramInventory program_inventory;
  std::vector<ConSanPerturbationCandidate> candidates;
  bool analysis_succeeded = false;
};

/// Rebuild the pristine synchronization semantics and perturbation candidates
/// needed to prove a committed perturbation. No perturbation plan is selected.
[[nodiscard]] ConSanPerturbationValidationInventory
rederive_consan_perturbation_validation_inventory(std::span<const uint8_t> original_image);

} // namespace rocjitsu
