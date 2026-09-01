// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_pipeline.h
/// @brief Cross-engine MOI resource-planning and lowering orchestration.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::vector<std::string>
summarize_moi_lowering(ConSanMoiEngine engine, bool modified,
                       std::span<const ConSanPatchKind> patch_kinds);

void publish_pending_moi_lowering_rejections(
    ConSanTransformArtifacts &result,
    std::optional<ConSanRegisterPlanReason> whole_transform_resource_failure = std::nullopt);

[[nodiscard]] ConSanMoiResourcePlanningResult solve_automatic_moi_exec_save_resources(
    MoiResourcePlanningState &state, const ConSanOptions &input,
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem,
    std::span<const ConSanMoiCandidate> candidates, const ConSanTransformArtifacts &artifacts);

void rebuild_moi_resource_plans(MoiResourcePlanningState &state, const ConSanRequest &request,
                                const BoundRuntimeResources &bound_resources,
                                const ConSanDebugOverrides &debug,
                                const ConSanMoiOperatingPoint &point,
                                const MoiObjectModeSemantics &mode_semantics,
                                std::span<const ConSanMoiCandidate> candidates,
                                ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
