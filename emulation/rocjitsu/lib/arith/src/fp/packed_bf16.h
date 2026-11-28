// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native Loose packed BF16 arithmetic backends; Compatibility keeps compensated rounding.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class PackedBf16 { Add, Mul, Fma };

// Evaluate both halves with native FP32 arithmetic and nearest-even BF16 narrowing.
// Requires validated controls and x86-64-v3; preserve aliases and inactive lanes.
void packed_bf16_x86_64_v3(PackedBf16 op, bool saturate, uint32_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate both halves with native FP32 arithmetic and nearest-even BF16 narrowing.
// Requires validated controls and x86-64-v4; preserve aliases and inactive lanes.
void packed_bf16_x86_64_v4(PackedBf16 op, bool saturate, uint32_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
