// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed WMMA accumulator conversion, special-value encoding, and FP16
// integer-grid rounding. These rules differ from ordinary ALU conversions.

#pragma once

#include "common/bits.h"
#include <cstdint>

// Adapted from rocjitsu shared/dot_packed16.h.
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::packed16 {

/// @brief Narrow infinity or NaN bits using the packed WMMA payload policy.
template <bool Bf16> inline uint16_t special(uint32_t bits) {
  // Factor NaNs have a format-specific diagnostic payload on both cards.
  if (bits == 0xffc00a3d) {
    return Bf16 ? 0xfffd : 0xfe3d;
  }
  if constexpr (Bf16) {
    return uint16_t(bits >> 16);
  }
  return uint16_t(((bits >> 16) & 0x8000) | 0x7c00 | ((bits >> 13) & 1023));
}

/// @brief Round an aligned integer sum directly to F16 using nearest-even.
/// @details Rounding through FP32 first can change an F16 halfway case. BF16
/// instead truncates the architecture-specific FP32 result in the callers.
inline uint16_t pack_f16(int64_t units, int grid, bool frame_negative = false,
                         bool fp16_ovfl = false) {
  if (!units) {
    return 0;
  }
  constexpr int fraction_bits = 10;
  constexpr int bias = 15;
  constexpr uint16_t infinity = 0x7c00;
  const uint16_t sign = uint16_t((units < 0) != frame_negative) << 15;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  int top = int(rocjitsu::arith::bit_width(magnitude)) - 1;
  int exponent = top + grid;
  const int shift = exponent < 1 - bias ? 1 - bias - fraction_bits - grid : top - fraction_bits;
  uint64_t significand = shift <= 0 ? magnitude << -shift : shift >= 64 ? 0 : magnitude >> shift;
  if (shift > 0 && shift < 64) {
    const uint64_t tail = magnitude & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    significand += tail > half || (tail == half && (significand & 1));
  }
  if (exponent < 1 - bias) {
    return significand ? sign | uint16_t(significand) : 0;
  }
  if (significand == (uint64_t{1} << (fraction_bits + 1))) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > bias) {
    return sign | (fp16_ovfl ? 0x7bff : infinity);
  }
  return sign | uint16_t((exponent + bias) << fraction_bits) |
         uint16_t(significand & ((1u << fraction_bits) - 1));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::packed16
