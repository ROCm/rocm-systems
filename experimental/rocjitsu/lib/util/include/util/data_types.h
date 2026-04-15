// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DATA_TYPES_H_
#define UTIL_DATA_TYPES_H_

#include <bit>
#include <cmath>
#include <cstdint>

namespace util {

// ---------------------------------------------------------------------------
// IEEE 754 half-precision (FP16) conversion
// ---------------------------------------------------------------------------

/// @brief Convert a 16-bit IEEE 754 half-precision value to float.
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 16-bit IEEE 754 half-precision.
inline uint16_t f32_to_f16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 13) & 0x3FF;
  if (exp <= 0)
    return static_cast<uint16_t>(sign);
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

// ---------------------------------------------------------------------------
// BFloat16 (BF16) conversion
// ---------------------------------------------------------------------------

/// @brief Convert a 16-bit BFloat16 value to float.
inline float bf16_to_f32(uint16_t h) {
  uint32_t f = static_cast<uint32_t>(h) << 16;
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 16-bit BFloat16 (truncation, no rounding).
inline uint16_t f32_to_bf16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  return static_cast<uint16_t>(f >> 16);
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) conversion - 1 sign, 4 exponent, 3 mantissa bits
// ---------------------------------------------------------------------------

/// @brief Convert an 8-bit E4M3 FP8 value to float.
inline float fp8_e4m3_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 3) & 0xF;
  uint32_t mant = v & 0x7;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 15) {
    // E4M3: NaN when exp=15 and mant!=0; no infinity
    if (mant != 0)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
    // exp=15, mant=0 → max normal value
    return std::bit_cast<float>((sign << 31) | ((15 + 127 - 7) << 23) | (0 << 20));
  }
  if (exp == 0) {
    // Denormalized
    float result = std::ldexp(static_cast<float>(mant), -9);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 7) << 23) | (mant << 20);
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 8-bit E4M3 FP8 (truncation).
inline uint8_t f32_to_fp8_e4m3(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 7;
  uint32_t mant = (f >> 20) & 0x7;
  if (exp <= 0)
    return static_cast<uint8_t>(sign);
  if (exp >= 15)
    return static_cast<uint8_t>(sign | 0x7E); // max normal
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant);
}

// ---------------------------------------------------------------------------
// BF8 (E5M2) conversion - 1 sign, 5 exponent, 2 mantissa bits
// ---------------------------------------------------------------------------

/// @brief Convert an 8-bit E5M2 BF8 value to float.
inline float bf8_e5m2_to_f32(uint8_t v) {
  uint32_t sign = (v >> 7) & 1;
  uint32_t exp = (v >> 2) & 0x1F;
  uint32_t mant = v & 0x3;
  if (exp == 0 && mant == 0)
    return std::bit_cast<float>(sign << 31);
  if (exp == 31) {
    if (mant != 0)
      return std::bit_cast<float>((sign << 31) | 0x7FC00000u);
    return std::bit_cast<float>((sign << 31) | 0x7F800000u); // infinity
  }
  if (exp == 0) {
    float result = std::ldexp(static_cast<float>(mant), -16);
    return sign ? -result : result;
  }
  uint32_t f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 21);
  return std::bit_cast<float>(f);
}

/// @brief Convert a float to 8-bit E5M2 BF8 (truncation).
inline uint8_t f32_to_bf8_e5m2(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 24) & 0x80;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 21) & 0x3;
  if (exp <= 0)
    return static_cast<uint8_t>(sign);
  if (exp >= 31)
    return static_cast<uint8_t>(sign | 0x7C | (mant ? 0x1 : 0));
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | mant);
}

} // namespace util

#endif // UTIL_DATA_TYPES_H_
