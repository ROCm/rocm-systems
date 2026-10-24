// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Width-aware integer extension, bit counting, and bit reversal.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <bit>
#include <stdint.h>

namespace rocjitsu::arith {

// Decode the low Bits bits as a signed or unsigned integer, discarding higher bits.
template <int Bits, bool Signed> int64_t extend_integer(uint32_t value) {
  static_assert(Bits > 0 && Bits <= 32);
  value &= UINT32_MAX >> (32 - Bits);
  if constexpr (Signed) {
    constexpr uint32_t sign = 1U << (Bits - 1);
    if constexpr (Bits <= 16) {
      return int64_t(value ^ sign) - sign;
    } else {
      // Keep the wider-field form that avoids slower GCC high-product lowering.
      return int64_t(value) - ((value & sign) ? INT64_C(1) << Bits : 0);
    }
  } else {
    return value;
  }
}

// Number of set bits in an unsigned 32- or 64-bit word.
template <typename T> uint32_t bit_population(T x) {
  x -= (x >> 1) & T(0x5555555555555555ULL);
  x = (x & T(0x3333333333333333ULL)) + ((x >> 2) & T(0x3333333333333333ULL));
  x = (x + (x >> 4)) & T(0x0f0f0f0f0f0f0f0fULL);
  x += x >> 8;
  x += x >> 16;
  if constexpr (sizeof(T) == 8) {
    x += x >> 32;
  }
  return uint32_t(x) & (sizeof(T) == 8 ? 127 : 63);
}

// Leading or trailing zero count; a zero word returns UINT32_MAX.
template <bool Trailing, typename T> uint32_t bit_count_zero(T a) {
  if (!a) {
    return UINT32_MAX;
  }
  return Trailing ? std::countr_zero(a) : std::countl_zero(a);
}

// Reverse all bits of an unsigned 32- or 64-bit word.
template <typename T> T bit_reverse(T x) {
  const T m1 = T(0x5555555555555555ULL);
  const T m2 = T(0x3333333333333333ULL);
  const T m4 = T(0x0f0f0f0f0f0f0f0fULL);
  const T m8 = T(0x00ff00ff00ff00ffULL);
  x = ((x >> 1) & m1) | ((x & m1) << 1);
  x = ((x >> 2) & m2) | ((x & m2) << 2);
  x = ((x >> 4) & m4) | ((x & m4) << 4);
  x = ((x >> 8) & m8) | ((x & m8) << 8);
  if constexpr (sizeof(T) == 8) {
    const T m16 = 0x0000ffff0000ffffULL;
    x = ((x >> 16) & m16) | ((x & m16) << 16);
    return (x >> 32) | (x << 32);
  } else {
    return (x >> 16) | (x << 16);
  }
}

} // namespace rocjitsu::arith
