// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file dot_packed16.h
/// @brief Integer conversion and special-value handling for packed WMMA.

#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu::dot_packed16 {

/// @brief Widen a packed value to FP32 bits without host FP operations.
/// @details Preserves signaling NaN payloads without quieting them.
template <bool Bf16> inline uint32_t widen(uint16_t bits) {
  if constexpr (Bf16)
    return uint32_t(bits) << 16;
  const uint32_t sign = uint32_t(bits & 0x8000) << 16;
  const uint32_t exponent = (bits >> 10) & 31;
  const uint32_t fraction = bits & 1023;
  if (exponent == 31)
    return sign | 0x7f800000u | (fraction << 13);
  if (exponent)
    return sign | ((exponent + 112) << 23) | (fraction << 13);
  if (!fraction)
    return sign;
  const unsigned shift = std::countl_zero(fraction) - 21;
  return sign | ((113 - shift) << 23) | (((fraction << shift) & 1023) << 13);
}

/// @brief Narrow infinity or NaN bits using the packed WMMA payload policy.
template <bool Bf16> inline uint16_t special(uint32_t bits) {
  // Factor NaNs have a format-specific diagnostic payload on both cards.
  if (bits == 0xffc00a3d)
    return Bf16 ? 0xfffd : 0xfe3d;
  if constexpr (Bf16)
    return uint16_t(bits >> 16);
  return uint16_t(((bits >> 16) & 0x8000) | 0x7c00 | ((bits >> 13) & 1023));
}

/// @brief Round an aligned integer sum directly to F16 using nearest-even.
/// @details Rounding through FP32 first can change an F16 halfway case. BF16
/// instead truncates the architecture-specific FP32 result in the callers.
inline uint16_t pack_f16(int64_t units, int grid, bool frame_negative = false,
                         bool fp16_ovfl = false) {
  if (!units)
    return 0;
  constexpr int fraction_bits = 10;
  constexpr int bias = 15;
  constexpr uint16_t infinity = 0x7c00;
  const uint16_t sign = uint16_t((units < 0) != frame_negative) << 15;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = int(std::bit_width(magnitude)) - 1;
  int exponent = top + grid;
  const int shift = exponent < 1 - bias ? 1 - bias - fraction_bits - grid : top - fraction_bits;
  uint64_t significand = shift <= 0 ? magnitude << -shift : shift >= 64 ? 0 : magnitude >> shift;
  if (shift > 0 && shift < 64) {
    const uint64_t tail = magnitude & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    significand += tail > half || (tail == half && (significand & 1));
  }
  if (exponent < 1 - bias)
    return significand ? sign | uint16_t(significand) : 0;
  if (significand == (uint64_t{1} << (fraction_bits + 1))) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > bias)
    return sign | (fp16_ovfl ? 0x7bff : infinity);
  return sign | uint16_t((exponent + bias) << fraction_bits) |
         uint16_t(significand & ((1u << fraction_bits) - 1));
}

/// @brief Broadcast an inline constant's 16-bit encoding into both halves.
/// @details Integer constants use raw low bits; float constants are narrowed by
/// format. Selectors 240 through 248 contain only finite normal FP32 constants,
/// so their F16 conversion below assumes an implicit leading significand bit.
/// This helper is not a general converter for literal or SGPR sources.
template <bool Bf16> inline uint32_t inline_word(uint32_t raw, uint16_t selector) {
  uint16_t bits = uint16_t(raw);
  if (selector >= 240 && selector <= 248) {
    if constexpr (Bf16)
      bits = uint16_t(raw >> 16);
    else {
      const int64_t significand = (raw & 0x7fffff) | 0x800000;
      bits = pack_f16((raw >> 31) ? -significand : significand, int((raw >> 23) & 255) - 127 - 23);
    }
  }
  return uint32_t(bits) | (uint32_t(bits) << 16);
}
} // namespace rocjitsu::amdgpu::dot_packed16
