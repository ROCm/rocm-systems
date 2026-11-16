// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Source layouts and ISA-specific backends for scaled narrow-float conversion.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
enum class ScaledNarrowSource { F32, F16, Bf16 };
enum class ScaledNarrowDestination { Fp8, Bf8, Fp4, Fp6, Bf6 };

// Scale and narrow a validated wave on the named ISA. Keep inactive lanes,
// unselected destination bytes and lanes with scale exponent 255 unchanged.
// upper_a supplies the second FP32 source for FP4 stochastic rounding; all
// other forms ignore it. shift selects the destination byte or halfword.
template <ScaledNarrowDestination Dst, bool Stochastic>
void scaled_narrow_x86_64_v3(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                             bool saturate, bool flush_input, int shift, uint32_t *d,
                             const uint32_t *a, const uint32_t *b, const uint32_t *scale,
                             const uint32_t *upper_a);
template <ScaledNarrowDestination Dst, bool Stochastic>
void scaled_narrow_x86_64_v4(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                             bool saturate, bool flush_input, int shift, uint32_t *d,
                             const uint32_t *a, const uint32_t *b, const uint32_t *scale,
                             const uint32_t *upper_a);
// Narrow 32 gfx950 or 8/16 gfx1250 elements, preserving inactive lanes.
// FullScale selects gfx1250 division; gfx950 special-scale lanes preserve D.
template <ScaledNarrowDestination Dst, bool Stochastic, bool FullScale = false>
void wide_scaled_narrow_x86_64_v3(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *scale, bool saturate,
                                  bool flush_input);
template <ScaledNarrowDestination Dst, bool Stochastic, bool FullScale = false>
void wide_scaled_narrow_x86_64_v4(int lanes, uint64_t exec_mask, ScaledNarrowSource source,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *scale, bool saturate,
                                  bool flush_input);
} // namespace rocjitsu::arith
