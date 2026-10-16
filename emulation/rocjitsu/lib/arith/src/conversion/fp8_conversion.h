// SPDX-License-Identifier: MIT

// FP8/BF8 widening to FP32, including packed pairs.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Widen FP8/BF8 factors to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v3(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a);

// Widen FP8/BF8 factors to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v4(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a);

} // namespace rocjitsu::arith
