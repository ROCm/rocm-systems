// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Truncating FP32-to-FP16 conversion with the GPU NaN/overflow rules.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Round toward zero to FP16, saturating finite overflow and quieting NaNs.
inline uint16_t half_rtz(uint32_t raw) {
  // Adapted from rocjitsu's util::f32_to_f16_rtz; RDNA4 quiets source NaNs.
  uint32_t sign = (raw >> 16) & 0x8000;
  int exponent = (raw >> 23) & 255;
  uint32_t fraction = raw & 0x7fffff;
  if (exponent == 255) {
    return uint16_t(sign | 0x7c00 | (fraction ? (fraction >> 13) | 0x200 : 0));
  }
  int adjusted = exponent - 112;
  if (adjusted <= 0) {
    return uint16_t(sign | (adjusted < -10 ? 0 : (fraction | 0x800000) >> (14 - adjusted)));
  }
  if (adjusted >= 31) {
    return uint16_t(sign | 0x7bff);
  }
  return uint16_t(sign | (unsigned(adjusted) << 10) | (fraction >> 13));
}

} // namespace rocjitsu::arith
