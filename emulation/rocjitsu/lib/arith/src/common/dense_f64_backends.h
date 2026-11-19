// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense FP64 matrix backend declarations, with a separate symbol for each ISA domain.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
// Evaluate a whole FP64 tile with validated sign controls and a compile-time layout.
// Bits 0/1/2 negate A/B/C; bit 3 takes C's absolute value before its negation.
template <int M, int Batch, int Lanes = 64>
void dense_f64_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);
template <int M, int Batch, int Lanes = 64>
void dense_f64_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);
} // namespace rocjitsu::arith
