// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exact mixed-precision FMA operand preparation and FP16 result handling.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_integer.h"
#include "fp/fp64.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Return the FP16 FMA result when at least one FP32 input is nonfinite.
inline uint16_t mixed_fma_special(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;
  if ((ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000)) {
    return 0xfe00;
  }
  if (ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000) {
    uint32_t nan = ma > 0x7f800000 ? a : mb > 0x7f800000 ? b : c;
    return uint16_t(((nan >> 16) & 0x8000) | 0x7e00 | ((nan >> 13) & 1023));
  }
  if (ma == 0x7f800000 || mb == 0x7f800000) {
    uint32_t sign = (a ^ b) & 0x80000000;
    if (mc == 0x7f800000 && ((sign ^ c) & 0x80000000)) {
      return 0xfe00;
    }
    return uint16_t((sign >> 16) | 0x7c00);
  }
  return uint16_t(((c >> 16) & 0x8000) | 0x7c00);
}

} // namespace rocjitsu::arith
