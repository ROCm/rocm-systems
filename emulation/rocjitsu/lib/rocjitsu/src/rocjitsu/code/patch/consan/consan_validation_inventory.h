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

/// Exact immutable facts rederived once from the pristine code object for
/// mutation and perturbation proof.
struct ConSanPristineValidationInventory {
  ProgramInventory program_inventory;
  std::vector<ConSanFaultSite> fault_sites;
  std::vector<ConSanPerturbationCandidate> perturbation_candidates;
  bool analysis_succeeded = false;

  [[nodiscard]] ConSanFaultSelectionView fault_selection() const {
    return {.program_inventory = program_inventory, .fault_sites = fault_sites};
  }
};

/// Rebuild every pristine semantic fact needed by final mutation and
/// perturbation proof. Extended barrier pairing is enabled when mutation
/// semantics must be reproduced. No mutation bytes are applied and no
/// perturbation plan is selected.
[[nodiscard]] ConSanPristineValidationInventory rederive_consan_pristine_validation_inventory(
    std::span<const uint8_t> original_image, bool require_mutation_semantics);

} // namespace rocjitsu
