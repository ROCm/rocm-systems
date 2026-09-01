// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled.h
/// @brief Compiled Sampled engine lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] uint64_t
sampled_reserved_barrier_island_count(const ConSanObservationPlan &observation_plan,
                                      const ConSanRequest &request, const TransformPolicy &policy);

[[nodiscard]] uint32_t sampled_reserved_atomic_island_count(
    const ConSanObservationPlan &observation_plan, const ConSanRequest &request,
    const BoundRuntimeResources &resources, const TransformPolicy &policy);

void try_apply_direct_sampled_watchpoint_patch(std::span<const uint8_t> bytes,
                                               const MoiOptions &options, rj_code_arch_t arch,
                                               MoiResourcePlanningState &resource_state,
                                               std::span<const ConSanMoiCandidate> admitted,
                                               ConSanTransformArtifacts &result);

void try_apply_sampled_barrier_sync_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options,
                                          const ConSanMoiOperatingPoint &operating_point,
                                          rj_code_arch_t arch,
                                          MoiResourcePlanningState &resource_state,
                                          std::span<const ConSanMoiCandidate> admitted,
                                          ConSanTransformArtifacts &result);

void try_apply_sampled_atomic_sync_patch(std::span<const uint8_t> bytes,
                                         const ConSanOptions &options,
                                         const ConSanMoiOperatingPoint &operating_point,
                                         rj_code_arch_t arch,
                                         MoiResourcePlanningState &resource_state,
                                         ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
