// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Basic FP16/BF16 storage conversions for graphics, memory and operand staging.
// Instruction arithmetic and mode-dependent conversions belong to lib/arith/.

#ifndef UTIL_DATA_TYPES_H_
#define UTIL_DATA_TYPES_H_

#include <bit>
#include <cstdint>

namespace util {

// ---- IEEE 754 half-precision (FP16) ----

// Widen an IEEE half exactly, preserving NaN sign and payload bits.
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

/// @brief Round F32 to F16 with nearest-even rounding, preserving retained NaN payload bits.
/// @details If truncation discards the entire payload, keep bit 0 to avoid producing infinity.
inline uint16_t f32_to_f16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  uint32_t f_exp = (f >> 23) & 0xFF;
  uint32_t f_mant = f & 0x7FFFFF;

  if (f_exp == 0xFF) {
    if (f_mant) {
      const uint32_t payload = f_mant >> 13;
      return static_cast<uint16_t>(sign | 0x7C00 | (payload != 0 ? payload : 1));
    }
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    uint32_t mant = f_mant | 0x800000;
    int shift = 14 - exp;
    uint32_t round_bit = (mant >> (shift - 1)) & 1;
    uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) ? 1 : 0;
    uint32_t result = mant >> shift;
    result += round_bit & (sticky | (result & 1));
    return static_cast<uint16_t>(sign | result);
  }

  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  uint32_t round_bit = (f_mant >> 12) & 1;
  uint32_t sticky = (f_mant & 0xFFF) ? 1 : 0;
  uint32_t mant = f_mant >> 13;
  mant += round_bit & (sticky | (mant & 1));
  if (mant > 0x3FF) {
    mant = 0;
    exp += 1;
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

/// @brief Truncate F32 to F16 with the same NaN payload policy as f32_to_f16.
inline uint16_t f32_to_f16_rtz(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  uint32_t sign = (f >> 16) & 0x8000;
  uint32_t f_exp = (f >> 23) & 0xFF;
  uint32_t f_mant = f & 0x7FFFFF;

  if (f_exp == 0xFF) {
    if (f_mant) {
      const uint32_t payload = f_mant >> 13;
      return static_cast<uint16_t>(sign | 0x7C00 | (payload != 0 ? payload : 1));
    }
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  int32_t exp = static_cast<int32_t>(f_exp) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    uint32_t mant = f_mant | 0x800000;
    int shift = 14 - exp;
    return static_cast<uint16_t>(sign | (mant >> shift));
  }

  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7BFF);
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (f_mant >> 13));
}

// ---- BFloat16 (BF16) ----

// Widen a BF16 encoding exactly, preserving NaN sign and payload bits.
inline float bf16_to_f32(uint16_t h) {
  uint32_t f = static_cast<uint32_t>(h) << 16;
  return std::bit_cast<float>(f);
}

// Return the high 16 FP32 bits without rounding.
inline uint16_t f32_to_bf16(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  return static_cast<uint16_t>(f >> 16);
}

/// @brief Convert a float to 16-bit BFloat16 with round-to-nearest-even.
inline uint16_t f32_to_bf16_rne(float val) {
  uint32_t f = std::bit_cast<uint32_t>(val);
  if ((f & 0x7f800000u) != 0x7f800000u) {
    f += 0x7fffu + ((f >> 16) & 1u);
  } else if (f & 0xffffu) {
    f |= 0x10000u;
  }
  return static_cast<uint16_t>(f >> 16);
}

} // namespace util

#endif // UTIL_DATA_TYPES_H_
