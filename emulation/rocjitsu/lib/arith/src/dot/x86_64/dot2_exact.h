// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD interfaces for bit-exact gfx11/gfx12 FP16/BF16 DOT2 SIMD kernels.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate dot2 exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_exact_x86_64_v3(bool bf16, bool gfx11, uint32_t exec_mask, uint32_t modifiers,
                          uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate dot2 exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_exact_x86_64_v4(bool bf16, bool gfx11, uint32_t exec_mask, uint32_t modifiers,
                          uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
