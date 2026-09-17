// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider.h
/// @brief SuperCollider planning, placement, and lowering boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <string_view>

namespace rocjitsu {
class AmdGpuCodeObject;
}

namespace rocjitsu::consan {

void publish_unplaced_supercollider_access_rejections(CoverageLedger &coverage,
                                                      std::vector<std::string> &errors);

void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object,
                                         const TargetProfile &target, const Options &options,
                                         TransformArtifacts &result);
void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object,
                                     const TargetProfile &target, const Options &options,
                                     TransformArtifacts &result);

} // namespace rocjitsu::consan
