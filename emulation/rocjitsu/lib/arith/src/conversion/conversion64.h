// SPDX-License-Identifier: MIT

// Conversions between FP64 and FP32 or 32-bit integers in split VGPR layouts.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Conversion64 {
  SignedToDouble,
  UnsignedToDouble,
  DoubleToSigned,
  DoubleToUnsigned,
  FloatToDouble,
  DoubleToFloat
};

// Convert FP64, FP32, or 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion64 Op>
void conversion64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a);

// Convert FP64, FP32, or 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion64 Op>
void conversion64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a);

} // namespace rocjitsu::arith
