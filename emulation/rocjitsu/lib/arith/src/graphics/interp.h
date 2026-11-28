// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend interfaces for FP16/FP32 interpolation arithmetic.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Each backend commits whole quads. Exact callers receive remaining EXEC at the
// first block needing guest NaN handling; Loose computes native nonfinite results.

// Loose FP32 quad interpolation: P10 reads quad A[1]/C[0], P2 reads quad A[2].
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
uint32_t interp32_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose FP32 quad interpolation: P10 reads quad A[1]/C[0], P2 reads quad A[2].
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
uint32_t interp32_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Native Loose mixed interpolation; no exact reconstruction or nonfinite repair.
// Preserve the same quad routing, modifiers, inactive lanes and destination halves.
template <bool P2, bool Rtz>
void interp16_loose_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose mixed FP16/FP32 quad interpolation, with RNE or RTZ narrowing.
// P10 writes FP32; P2 preserves the unselected FP16 destination half.
// Source lanes may be inactive; preserve inactive destinations and input aliases.
// Requires the named CPU features and validated instruction controls.
template <bool P2, bool Rtz>
uint32_t interp16_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
