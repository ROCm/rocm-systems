// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 Wave32 FP16 fused multiply-add, with explicit EXEC and whole-VGPR aliases.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate loose FP16 FMA, correcting FP32 intermediates at FP16 midpoints.
static uint16_t loose_half_fma(float x, float y, float z, bool saturate) {
  float value = std::fma(x, y, z);
  const uint32_t magnitude = as_bits(value) & 0x7fffffffU;
  const int exponent = int(magnitude >> 23);
  // Midpoints from half of the smallest subnormal through the overflow boundary.
  if (exponent >= 102 && exponent <= 142) {
    const int shift = exponent >= 113 ? 13 : 126 - exponent;
    const uint32_t significand = (magnitude & 0x7fffffU) | 0x800000U;
    if ((significand & ((1U << shift) - 1)) == (1U << (shift - 1))) {
      const double precise = std::fma(double(x), double(y), double(z));
      // One FP32 step restores which side of the half midpoint the result lies on.
      if (precise != double(value)) {
        value = std::nextafter(value, precise > value ? INFINITY : -INFINITY);
      }
    }
  }
  return float_to_f16(value, saturate);
}

} // namespace rocjitsu::arith

// Evaluate the low halfword and preserve the high half and inactive lanes.
// Loose semantics do not access the FP status pointer.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  (void)excp_flag_user;
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    if (rocjitsu::arith::is_exact(flags)) {
      result[lane] =
          rocjitsu::arith::half_fma_value(uint16_t(a[0][lane]), uint16_t(b[0][lane]),
                                          uint16_t(c[0][lane]), bool(flags & RJ_ARITH_FP16_OVFL));
    } else {
      float x = rocjitsu::arith::f16_to_float(uint16_t(a[0][lane]));
      float y = rocjitsu::arith::f16_to_float(uint16_t(b[0][lane]));
      float z = rocjitsu::arith::f16_to_float(uint16_t(c[0][lane]));
      result[lane] = rocjitsu::arith::loose_half_fma(x, y, z, flags & RJ_ARITH_FP16_OVFL);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & 0xffff0000U) | result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
