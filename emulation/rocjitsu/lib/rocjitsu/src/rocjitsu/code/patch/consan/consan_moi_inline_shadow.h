// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_inline_shadow.h
/// @brief Compiled InlineShadow engine lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
                                   std::span<const ConSanMoiCandidate> admitted,
                                   const MoiObjectModeSemantics &semantics,
                                   ConSanTransformArtifacts &result);

void try_apply_inline_atomic_ordering_patch(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            rj_code_arch_t arch,
                                            const MoiObjectModeSemantics &semantics,
                                            ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
