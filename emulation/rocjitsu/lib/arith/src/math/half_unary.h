// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 unary math, integral rounding, and special-value policies.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "math/unary.h"

#include <stdint.h>

namespace rocjitsu::arith {

// RCP/RSQ/SQRT use the captured FP32 approximation followed by half narrowing.
template <Unary Op>
constexpr bool half_unary_sfu = Op == Unary::Rcp || Op == Unary::Rsq || Op == Unary::Sqrt;

// Loose FP16 unary arithmetic through FP32, with half selection and modifiers.
// Preserve inactive lanes and the unselected half; D may alias A.
// Requires AVX2 and validated instruction controls.
template <Unary Op, bool GuestDenormals = true>
void half_unary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a);

// Exact FP16 RCP/RSQ/SQRT using the inherited approximation and half narrowing.
// Preserve inactive lanes and the unselected half; D may alias A.
// Requires AVX2 and validated instruction controls.
template <Unary Op>
void half_sfu_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, bool quiet_snan);

// Exact FP16 TRUNC/CEIL/FLOOR/RNDNE, with half selection and modifiers.
// Preserve inactive lanes and the unselected half; D may alias A.
// Requires x86-64-v4 and validated instruction controls.
template <Unary Op>
void half_round_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                const uint32_t *a);

} // namespace rocjitsu::arith
