// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_resource_planning.h
/// @brief Cross-mode ConSan resource-planning and lowering orchestration.

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"

namespace rocjitsu::consan::detail {

[[nodiscard]] std::vector<std::string> summarize_lowering(bool modified,
                                                          std::span<const PatchKind> patch_kinds);

[[nodiscard]] ResourcePlanningResult
solve_automatic_exec_save_resources(ResourcePlanningState &state, const DebugOverrides &debug,
                                    const OperatingPoint &base, const ResourceProblem &problem);

[[nodiscard]] ResourcePlanningResult plan_resources(ResourcePlanningState &state,
                                                    const DebugOverrides &debug,
                                                    const OperatingPoint &point,
                                                    const ResourceProblem &problem);

} // namespace rocjitsu::consan::detail
