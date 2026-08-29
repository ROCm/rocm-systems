// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay.h
/// @brief Record/Replay engine lowering entry points.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu::consan_moi_impl {

void try_apply_atomic_record_patch(std::span<const uint8_t> bytes, const MoiOptions &options,
                                   rj_code_arch_t arch, ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
