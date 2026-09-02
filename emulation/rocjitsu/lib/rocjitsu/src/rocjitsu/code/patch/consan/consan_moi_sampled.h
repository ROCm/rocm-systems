// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled.h
/// @brief Compiled Sampled engine lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool append_sampled_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    uint64_t descriptor_file_offset, bool automatic_private_epoch,
    const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
    const ConSanMoiPrivateStateLayout &layout, rj_code_arch_t arch,
    std::vector<std::string> &errors);

void try_apply_direct_sampled_watchpoint_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, std::span<const ConSanMoiCandidate> admitted,
    const MoiObjectModeSemantics &mode_semantics, ConSanTransformArtifacts &result);

void try_apply_sampled_barrier_sync_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, std::span<const ConSanMoiCandidate> admitted,
    const MoiObjectModeSemantics &mode_semantics, ConSanTransformArtifacts &result);

void try_apply_sampled_atomic_sync_patch(std::span<const uint8_t> bytes,
                                         const ConSanOptions &options,
                                         const ConSanMoiOperatingPoint &operating_point,
                                         rj_code_arch_t arch,
                                         MoiResourcePlanningState &resource_state,
                                         const MoiObjectModeSemantics &mode_semantics,
                                         ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
