// SPDX-License-Identifier: MIT

// Backend interfaces for FP16/FP32 interpolation arithmetic.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate interp32 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c);

// Evaluate interp32 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c);

// Evaluate interp16 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2, bool Rtz>
void interp16_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith
