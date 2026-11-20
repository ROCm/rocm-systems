// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Truncating FP32-to-FP16 conversion with the GPU NaN/overflow rules.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Round toward zero to FP16, saturating finite overflow. Preserve a nonzero NaN
// payload, optionally setting its quiet bit.
template <bool QuietNaN = true> inline uint16_t half_rtz(uint32_t raw) {
  // Adapted from rocjitsu's util::f32_to_f16_rtz; RDNA4 quiets source NaNs.
  uint32_t sign = (raw >> 16) & 0x8000;
  int exponent = (raw >> 23) & 255;
  uint32_t fraction = raw & 0x7fffff;
  if (exponent == 255) {
    uint32_t payload = fraction >> 13;
    if (fraction) {
      payload = QuietNaN ? payload | 0x200 : (payload ? payload : 1);
    }
    return uint16_t(sign | 0x7c00 | payload);
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

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
