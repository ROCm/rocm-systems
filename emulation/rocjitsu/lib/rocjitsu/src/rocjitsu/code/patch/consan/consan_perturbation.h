// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_perturbation.h
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

void build_perturbation_candidate_inventory(const ProgramInventory &program_inventory,
                                            ConSanPerturbationPlanningState &planning);

[[nodiscard]] std::string
consan_perturbation_candidate_identity(const ProgramInventory &program_inventory,
                                       const ConSanPerturbationCandidate &candidate);

void build_perturbation_plan(const ProgramInventory &program_inventory,
                             const ConSanRequest &request, const MutationRequest &mutation,
                             const ConSanDebugOverrides &debug,
                             ConSanPerturbationPlanningState &planning, ConSanMutationTally &tally,
                             std::vector<std::string> &errors,
                             std::span<const ConSanPerturbationPlan> carried_plans = {});

void try_apply_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                    const TransformPolicy &transform_policy,
                                    const ConSanPerturbationPlanningState &planning,
                                    ConSanTransformArtifacts &result);

} // namespace rocjitsu
