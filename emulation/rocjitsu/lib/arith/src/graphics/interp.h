// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend interfaces for FP16/FP32 interpolation arithmetic.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Loose FP32 quad interpolation: P10 reads quad A[1]/C[0], P2 reads quad A[2].
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c);

// Loose FP32 quad interpolation: P10 reads quad A[1]/C[0], P2 reads quad A[2].
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c);

// Loose mixed FP16/FP32 quad interpolation, with RNE or RTZ narrowing.
// P10 writes FP32; P2 preserves the unselected FP16 destination half.
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2, bool Rtz>
void interp16_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
