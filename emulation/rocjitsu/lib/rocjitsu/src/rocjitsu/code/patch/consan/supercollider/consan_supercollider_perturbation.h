// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_perturbation.h
/// @brief Typed SuperCollider perturbation planning and mutation boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
class AmdGpuCodeObject;
}

namespace rocjitsu::consan {

void build_supercollider_perturbation_candidate_inventory(
    const ProgramInventory &program_inventory, SuperColliderPerturbationPlanningState &planning);

[[nodiscard]] std::string
supercollider_perturbation_candidate_identity(const ProgramInventory &program_inventory,
                                              const SuperColliderPerturbationCandidate &candidate);

void build_supercollider_perturbation_plan(
    const ProgramInventory &program_inventory, const Request &request,
    const MutationRequest &mutation, const DebugOverrides &debug,
    SuperColliderPerturbationPlanningState &planning, MutationTally &tally,
    std::vector<std::string> &errors,
    std::span<const SuperColliderPerturbationPlan> carried_plans = {});

[[nodiscard]] bool
stage_supercollider_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                         const SuperColliderPerturbationPlanningState &planning,
                                         TransformArtifacts &result);

} // namespace rocjitsu::consan
