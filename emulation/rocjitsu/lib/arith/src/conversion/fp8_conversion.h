// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP8/BF8 widening to FP32 or FP16, including packed pairs.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Encoding and NaN policy inherited from each target conversion instruction.
enum class Fp8WidenPolicy { Gfx12, Gfx942, Gfx950, Gfx1250, E5M3 };

// Widen FP8/BF8 factors to the selected FP32/FP16 destination format. Preserve destination lanes
// outside exec_mask. Half selects FP16 output; single-half writes preserve the other destination
// half. Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed, bool Half = false>
void fp8_conversion_x86_64_v3(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a, Fp8WidenPolicy policy, uint64_t flags = 0,
                              uint32_t mode = 0);

// Widen FP8/BF8 factors to the selected FP32/FP16 destination format. Preserve destination lanes
// outside exec_mask. Half selects FP16 output; single-half writes preserve the other destination
// half. Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed, bool Half = false>
void fp8_conversion_x86_64_v4(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a, Fp8WidenPolicy policy, uint64_t flags = 0,
                              uint32_t mode = 0);

} // namespace rocjitsu::arith
