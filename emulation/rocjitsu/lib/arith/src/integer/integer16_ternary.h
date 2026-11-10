// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 16-bit multiply-add and three-input integer selections.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Integer16Ternary { Mad, Min, Max, Median };

// Evaluate three-input 16-bit integer arithmetic. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Signed, Integer16Ternary Op = Integer16Ternary::Mad, bool Packed = true>
void integer16_ternary_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
