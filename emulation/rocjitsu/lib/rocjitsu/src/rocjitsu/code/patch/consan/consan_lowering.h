// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering.h
/// @brief Typed output boundary for ConSan's remaining compatibility lowerer.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

namespace rocjitsu {

/// Lower one code object to production-owned static artifacts. Mutable working
/// state cannot cross this boundary while the option input is decomposed.
[[nodiscard]] ConSanTransformArtifacts lower_consan(std::span<const uint8_t> code_object_bytes,
                                                    const ConSanOptions &options);

} // namespace rocjitsu
