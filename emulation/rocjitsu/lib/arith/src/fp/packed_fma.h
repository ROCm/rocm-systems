// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP16 FMA/FMAC with independent low/high operand selectors.
// Backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate paired FP16 fused multiply-adds. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Accumulate>
void packed_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Compute both packed halves with nearest guest rounding. Return false without writes
// for directed rounding or requested status unless CLAMP suppresses it on both halves.
// Requires validated controls and AVX2; preserve inactive lanes and allow D/input aliasing.
bool packed_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *state, bool rdna3 = false);

// Compute both packed halves with the FP16 exact backend's validated status domain.
// Preserve inactive lanes and allow D to alias either input.
void packed_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *state, bool rdna3 = false);

} // namespace rocjitsu::arith
