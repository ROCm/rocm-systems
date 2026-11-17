// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Format selectors and ISA-specific backends for scaled widening.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
enum class ScaledSource { Fp4, Fp8, Bf8, Fp6, Bf6 };
enum class ScaledDestination { F32, F16, Bf16 };

// Convert one wave with validated controls and named CPU features. Keep inactive
// lanes and unselected destination halves, and permit whole-register aliases.
void scaled_widen_x86_64_v3(int lanes, uint64_t exec_mask, ScaledSource source,
                            ScaledDestination destination, bool packed, bool saturate, int shift,
                            int destination_shift, uint32_t *const *d, const uint32_t *a,
                            const uint32_t *b);
void scaled_widen_x86_64_v4(int lanes, uint64_t exec_mask, ScaledSource source,
                            ScaledDestination destination, bool packed, bool saturate, int shift,
                            int destination_shift, uint32_t *const *d, const uint32_t *a,
                            const uint32_t *b);
// Expand packed narrow inputs: gfx950 FP6/BF6 uses 32 elements; ByteScale
// selects gfx1250 E8M0 scaling, with 16 FP6/BF6 or 8 FP4/FP8/BF8 elements. Preserve
// inactive lanes; only gfx950 preserves D for scale 255. scale_shift selects B bits.
template <ScaledSource Source, bool ByteScale>
void wide_scaled_widen_x86_64_v3(int lanes, uint64_t exec_mask, ScaledDestination destination,
                                 bool saturate, int scale_shift, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *b);
template <ScaledSource Source, bool ByteScale>
void wide_scaled_widen_x86_64_v4(int lanes, uint64_t exec_mask, ScaledDestination destination,
                                 bool saturate, int scale_shift, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *b);
} // namespace rocjitsu::arith
