// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_planning.h
/// @brief Pure fault-mutation planning contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_fault_selection.h"

#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// Complete immutable input consumed by validation-only fault planning.
///
/// Planning may inspect semantic selection facts and barrier-move destinations.
/// It must not inspect lowering state, emitted patches, resource choices, or
/// transformation diagnostics.
struct ConSanFaultPlanningInput {
  ConSanFaultSelectionView selection;
  std::span<const ConSanBarrierMoveDestination> barrier_move_destinations;
  bool prior_attempt_unsupported = false;
};

/// Owned product of one fault-planning attempt.
///
/// The composition layer decides how to publish this product into its broader
/// transformation transaction. Planning itself neither owns nor mutates that
/// transaction.
struct ConSanFaultPlanningResult {
  std::vector<ConSanFaultMutationPlan> fault_plans;
  bool unsupported = false;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

[[nodiscard]] ConSanFaultPlanningResult
build_fault_mutation_plan(const ConSanOptions &options, rj_code_arch_t arch,
                          const ConSanFaultPlanningInput &input,
                          bool require_applicable_plan = false);

} // namespace rocjitsu
