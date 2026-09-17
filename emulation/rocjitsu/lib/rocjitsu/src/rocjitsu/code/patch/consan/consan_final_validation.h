// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_final_validation.h
/// @brief Private final-validation boundary for ConSan lowering.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>

namespace rocjitsu::consan {

/// Convert a staged transformation into its terminal result after independent
/// structural, semantic, resource, ABI, and mutation proof validation.
[[nodiscard]] TransformArtifacts finalize_result(TransformArtifacts result,
                                                 std::span<const uint8_t> original_bytes);

} // namespace rocjitsu::consan
