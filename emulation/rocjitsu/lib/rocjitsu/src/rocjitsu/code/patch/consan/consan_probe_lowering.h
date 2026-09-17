// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_probe_lowering.h
/// @brief Compiled ConSan mode lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_register_allocation.h"

namespace rocjitsu::consan::detail {

[[nodiscard]] bool append_private_owner_epoch_load(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, uint64_t descriptor_file_offset,
    bool automatic_private_epoch, const OwnerEpochVgprSources &owner_epoch_vgprs,
    const PrivateStateLayout &layout, rj_code_arch_t arch, std::vector<std::string> &errors);

void try_apply_direct_watchpoint_patch(std::span<const uint8_t> bytes, const Options &options,
                                       const OperatingPoint &operating_point, rj_code_arch_t arch,
                                       std::span<const Candidate> admitted,
                                       const ObjectModeSemantics &mode_semantics,
                                       TransformArtifacts &result);

void try_apply_barrier_sync_patch(std::span<const uint8_t> bytes, const Options &options,
                                  const OperatingPoint &operating_point, rj_code_arch_t arch,
                                  ResourcePlanningState &resource_state,
                                  std::span<const Candidate> admitted,
                                  const ObjectModeSemantics &mode_semantics,
                                  TransformArtifacts &result);

void try_apply_atomic_sync_patch(std::span<const uint8_t> bytes, const Options &options,
                                 const OperatingPoint &operating_point, rj_code_arch_t arch,
                                 ResourcePlanningState &resource_state,
                                 const ObjectModeSemantics &mode_semantics,
                                 TransformArtifacts &result);

} // namespace rocjitsu::consan::detail
