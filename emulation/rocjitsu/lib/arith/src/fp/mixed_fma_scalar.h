// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exact mixed-precision FMA operand preparation and FP16 result handling.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_integer.h"
#include "fp/half_fma_scalar.h"
#include "fp/mixed_fma_control.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode one mixed source, with its flags shifted to the A positions.
inline uint32_t mixed_fma_input(uint32_t raw, uint32_t mode, bool flush_input = false) {
  if (flush_input) {
    if (mode & RJ_ARITH_MIX_F16_A) {
      unsigned shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
      if (!((raw >> shift) & 0x7c00)) {
        raw &= ~(0x7fffU << shift);
      }
    } else if (!(raw & 0x7f800000)) {
      raw &= 0x80000000;
    }
  }
  if (mode & RJ_ARITH_MIX_F16_A) {
    raw = as_bits(f16_to_float(uint16_t(raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0))));
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    raw ^= 0x80000000;
  }
  return raw;
}

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

// Fuse FP32 inputs and round once to FP16, independently of host FP controls.
inline uint16_t mixed_fma_half_value(uint32_t a, uint32_t b, uint32_t c, bool clamp,
                                     uint64_t flags) {
  bool saturate = flags & RJ_ARITH_FP16_OVFL;
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    result = mixed_fma_special(a, b, c);
  } else {
    double value = as_double(fma_finite_round_odd(a, b, c));

    // Borrow the directed narrowing and exact-zero policy from half_fma_value
    // (rocjitsu fp_mode::fma_f16), retaining the integer FMA intermediate.
    const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
    if (value == 0) {
      const bool product_negative = (a ^ b) & 0x80000000;
      const bool matching_zeros = !(c & 0x7fffffff) && product_negative == bool(c & 0x80000000);
      value = (matching_zeros ? product_negative : rounding == 2) ? -0.0 : 0.0;
    }
    result = half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(value), saturate, rounding);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      const uint16_t magnitude = result & 0x7fff;
      const bool away = rounding == ((result & 0x8000) ? 2u : 1u);
      const double threshold = 0x1p-14 - (rounding == 0 ? 0x1p-26 : away ? 0x1p-25 : 0.0);
      if (magnitude < 0x400 ||
          (magnitude == 0x400 &&
           (__builtin_fabs(value) < threshold || (away && __builtin_fabs(value) == threshold)))) {
        result &= 0x8000;
      }
    }
  }
  const bool keep_nan = (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (result & 0x7fff) > 0x7c00;
  return clamp && !keep_nan ? half_fma_clamp(result) : result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
