// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// AVX2 and AVX-512 backend declarations for dispatch.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Numeric policies inherited from rocjitsu; Gfx12 also serves gfx1250.
// Every policy preserves destination bits outside the selected byte or half.
enum class Fp8NarrowPolicy { Gfx12, Fnuz, Ocp, E5M3 };

// Narrow FP16/FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Half selects FP16 inputs: packed forms read both halves of A and ignore B;
// stochastic forms read the selected A half and seed B. Requires validated controls.
template <bool Bf8, bool Stochastic, bool Half = false>
void fp8_narrow_x86_64_v3(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, Fp8NarrowPolicy policy);

// Narrow FP16/FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Half selects FP16 inputs: packed forms read both halves of A and ignore B;
// stochastic forms read the selected A half and seed B. Requires validated controls.
template <bool Bf8, bool Stochastic, bool Half = false>
void fp8_narrow_x86_64_v4(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, Fp8NarrowPolicy policy);

} // namespace rocjitsu::arith
