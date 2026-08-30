// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_barrier.h
/// @brief Compiled MOI barrier lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_replay.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] uint16_t inline_shadow_barrier_scratch_count(const ConSanRequest &request,
                                                           const BoundRuntimeResources &resources,
                                                           const ConSanMoiOperatingPoint &point);

void try_apply_inline_shadow_barrier_patch(std::span<const uint8_t> bytes,
                                           const MoiOptions &options, rj_code_arch_t arch,
                                           MoiResourcePlanningState &resource_state,
                                           ConSanTransformArtifacts &result);

void try_apply_record_replay_barrier_patch(std::span<const uint8_t> bytes,
                                           const MoiOptions &options, rj_code_arch_t arch,
                                           MoiResourcePlanningState &resource_state,
                                           const MoiRecordReplayAccessOutput &access_output,
                                           ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
