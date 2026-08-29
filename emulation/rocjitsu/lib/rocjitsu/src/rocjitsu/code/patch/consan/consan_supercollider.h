// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider.h
/// @brief SuperCollider planning, placement, and lowering boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <span>

namespace rocjitsu {

class AmdGpuCodeObject;

void publish_unplaced_sc_access_rejections(ConSanTransformArtifacts &result);

void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object,
                                         const ConSanTargetProfile &target,
                                         const ConSanOptions &options,
                                         ConSanTransformArtifacts &result);
void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object,
                                     const ConSanTargetProfile &target,
                                     const ConSanOptions &options, ConSanTransformArtifacts &result,
                                     std::span<const ByteRange> initial_reserved_ranges = {});
void try_apply_flat_trap_patch(const AmdGpuCodeObject &code_object,
                               const ConSanTargetProfile &target, const ConSanOptions &options,
                               ConSanTransformArtifacts &result);

} // namespace rocjitsu
