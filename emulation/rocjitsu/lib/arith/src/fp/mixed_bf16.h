// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend interface for mixed BF16/FP32 FMA and selected-half BF16 output.
#pragma once

#include "fp/mixed_fma.h"
#include <stdint.h>

namespace rocjitsu::arith {

// Return one decoded FP32-source lane under the inherited mixed BF16 policy.
// Baseline-compiled repair also serves exceptional lanes of SIMD kernels.
uint32_t mixed_bf16_value(MixedFma destination, uint64_t flags, bool clamp, uint32_t a, uint32_t b,
                          uint32_t c);

// Require validated controls and the named ISA. Preserve masked lanes and the
// unselected destination half; any source may alias D.
void mixed_bf16_x86_64_v3(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                          uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);
void mixed_bf16_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                          uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
