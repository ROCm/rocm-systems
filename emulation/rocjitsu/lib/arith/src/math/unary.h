// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 unary operation identities, policy traits, and backend declarations.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Unary { Trunc, Ceil, Rndne, Floor, Sqrt, Rcp, Rsq, Exp, Log, Fract, FrexpMant };

template <Unary Op>
inline constexpr bool unary_flushes_f32 = Op == Unary::Sqrt || Op == Unary::Rcp ||
                                          Op == Unary::Rsq || Op == Unary::Exp || Op == Unary::Log;

template <Unary Op>
inline constexpr bool unary_rounds_f32 =
    Op == Unary::Trunc || Op == Unary::Ceil || Op == Unary::Rndne || Op == Unary::Floor;

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Loose FP32 unary arithmetic with modifiers and guest flushing.
// Preserve inactive lanes and D/A aliases; produces no FP status.
// Requires AVX2 and validated instruction controls.
void unary_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                     const uint32_t *a, uint64_t flags);

// Integral rounding for all FP32 values, including NaN payload preservation.
void round_exact_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Exact FP32 TRUNC/CEIL/FLOOR/RNDNE, including signed zero and NaN policy.
// Preserve inactive lanes and D/A aliases. Requires x86-64-v4 and valid controls.
void round_exact_x86_64_v4(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags);
#endif

} // namespace rocjitsu::arith
