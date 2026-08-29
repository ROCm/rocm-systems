// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_inventory.h
/// @brief Narrow pristine-analysis inputs rederived for final validation.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

/// Rebuild the pristine barrier and fault inventory needed to prove an exact
/// mutation. No mutation bytes are applied by this dry-run analysis.
[[nodiscard]] ConSanTransformArtifacts
rederive_consan_mutation_validation_inventory(std::span<const uint8_t> original_image);

/// Pristine synchronization inventory plus the exact perturbation candidates
/// inspected while it was built.
struct ConSanPerturbationValidationInventory {
  ConSanTransformArtifacts artifacts;
  ConSanPerturbationPlanningState planning;
};

/// Rebuild the pristine synchronization semantics and perturbation candidates
/// needed to prove a committed perturbation.
[[nodiscard]] ConSanPerturbationValidationInventory
rederive_consan_perturbation_validation_inventory(std::span<const uint8_t> original_image);

} // namespace rocjitsu
