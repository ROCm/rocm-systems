// SPDX-License-Identifier: MIT

// Mixed FP16/FP32 FMA with FP32 or selected-halfword output.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class MixedFma { Float, Low, High };

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_exact_x86_64_v4(MixedFma destination, bool rdna3, uint64_t flags, uint32_t exec_mask,
                               uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c);

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_float_exact_x86_64_v3(bool rdna3, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                     uint32_t *d, const uint32_t *a, const uint32_t *b,
                                     const uint32_t *c);

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_float_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c);

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_float_guest_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Write the selected destination half, preserving the other half and inactive lanes.
void mixed_fma_half_x86_64_v3(bool high, bool saturate, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_half_guest_x86_64_v3(bool high, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                    uint32_t *d, const uint32_t *a, const uint32_t *b,
                                    const uint32_t *c);

} // namespace rocjitsu::arith
