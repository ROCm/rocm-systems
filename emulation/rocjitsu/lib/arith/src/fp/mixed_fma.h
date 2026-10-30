// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed FP16/FP32 FMA with FP32 or selected-halfword output.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class MixedFma { Float, Low, High };

// All backends require their named CPU features and validated controls; preserve inactive
// lanes and the unselected destination half. Inputs may alias D.
// Exact arithmetic for every guest rounding mode, with guest denormal and NaN policies.
void mixed_fma_exact_x86_64_v4(MixedFma destination, bool rdna3, uint64_t flags, uint32_t exec_mask,
                               uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c);

// Exact FP32 result with nearest guest rounding and guest denormal/NaN policies.
void mixed_fma_float_exact_x86_64_v3(bool rdna3, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                     uint32_t *d, const uint32_t *a, const uint32_t *b,
                                     const uint32_t *c);

// Loose FP32 result in host nearest rounding, with preserved denormals and default NaN clamp.
void mixed_fma_float_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c);

// Loose FP32 result in host nearest rounding, honoring guest denormal and NaN-clamp controls.
void mixed_fma_float_guest_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose FP16 result with nearest rounding and preserved denormals; saturate if requested.
void mixed_fma_half_x86_64_v3(bool high, bool saturate, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// FP16 result with guest rounding, denormal controls and saturation. Write the selected half.
void mixed_fma_half_guest_x86_64_v3(bool high, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                    uint32_t *d, const uint32_t *a, const uint32_t *b,
                                    const uint32_t *c);

} // namespace rocjitsu::arith
