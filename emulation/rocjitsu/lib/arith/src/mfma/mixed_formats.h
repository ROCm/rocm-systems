// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// OCP matrix format selection and packed subbyte widening, inherited from rocjitsu util.
#pragma once

#include "common/float_formats.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return the storage width of a validated format selector (0 through 4).
inline int matrix_format_bits(int format) { return format < 2 ? 8 : format < 4 ? 6 : 4; }

// Widen all finite FP6/BF6/FP4 codes, including negative zero, without a lookup table.
inline float matrix_decode_subbyte(uint32_t value, int format) {
  int fraction_bits = format == 2 ? 3 : format == 3 ? 2 : 1;
  int bits = matrix_format_bits(format);
  int bias = format == 3 ? 3 : 1;
  uint32_t sign = (value >> (bits - 1)) << 31;
  uint32_t magnitude = value & ((1U << (bits - 1)) - 1);
  uint32_t exponent = magnitude >> fraction_bits;
  uint32_t fraction = magnitude & ((1U << fraction_bits) - 1);
  if (!exponent) {
    float scale = format == 2 ? 0x1p-3f : format == 3 ? 0x1p-4f : 0x1p-1f;
    return as_float(as_bits(float(fraction) * scale) | sign);
  }
  return as_float(sign | ((exponent + 127 - bias) << 23) | (fraction << (23 - fraction_bits)));
}

// Read one packed element, including six-bit values straddling consecutive VGPRs.
inline float matrix_read_packed(const uint32_t *const *input, int lane, int element, int format) {
  int bits = matrix_format_bits(format), bit = element * bits;
  uint32_t value = input[bit / 32][lane] >> (bit % 32);
  if (bit % 32 + bits > 32) {
    value |= input[bit / 32 + 1][lane] << (32 - bit % 32);
  }
  value &= (1U << bits) - 1;
  if (format == 0) {
    return fp8_to_float<false>(uint8_t(value));
  }
  if (format == 1) {
    return fp8_to_float<true>(uint8_t(value));
  }
  return matrix_decode_subbyte(value, format);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
