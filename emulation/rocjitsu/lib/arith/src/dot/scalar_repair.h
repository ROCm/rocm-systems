// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline-ISA DOT repairs used by vector kernels for exceptional lanes.
// Only declarations belong in target-specific translation units.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Compute one gfx11 DOT2 or gfx12 DOT4 stage with FP16/BF16 factors and FP32 or packed C/D.
// The caller supplies two factors for gfx11, four for gfx12.
uint32_t repair_wmma_dot(const uint16_t *a, const uint16_t *b, uint32_t c, bool bf16, bool rdna3,
                         bool packed, bool saturate);

} // namespace rocjitsu::arith
