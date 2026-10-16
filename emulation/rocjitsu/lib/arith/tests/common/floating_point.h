// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deterministic floating-point inputs, bit representations, and exact/loose comparisons.

#pragma once

#include "rj_arith/rj_arith.h"

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// FP16 special-value bit patterns. The NaN is quiet with payload 1.
static const uint16_t kF16QuietNaNBits = 0x7e01;
static const uint16_t kF16PositiveInfinityBits = 0x7c00;

// BF16 special-value bit patterns. The NaN is quiet with payload 1.
static const uint16_t kBF16QuietNaNBits = 0x7fc1;
static const uint16_t kBF16PositiveInfinityBits = 0x7f80;

// FP32 special-value bit patterns. The NaN is quiet with payload 0x12345.
static const uint32_t kF32QuietNaNBits = 0x7fc12345U;
static const uint32_t kF32PositiveInfinityBits = 0x7f800000U;

// Copy the representation between equally sized trivially copyable types.
template <typename To, typename From> To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  static_assert(std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>);
  To to;
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

// Draw two finite FP16 values, using the high 16 bits of each engine result.
// Preserve sign and fractional bits while excluding the NaN/Inf exponent.
inline uint32_t random_f16_pair(std::minstd_rand &random) {
  uint32_t word = 0;
  for (int shift : {0, 16}) {
    uint32_t half = uint32_t(random()) >> 15;
    word |= ((half & 0x8000) | ((half & 0x7fff) % 0x7c00)) << shift;
  }
  return word;
}

// Draw two finite BF16 values from the high 16 engine bits, with unbiased
// exponents -8..7 and arbitrary signs/fractions. This keeps products finite.
inline uint32_t random_bf16_pair(std::minstd_rand &random) {
  uint32_t word = 0;
  for (int shift : {0, 16}) {
    uint32_t half = uint32_t(random()) >> 15;
    word |= ((half & 0x8000) | ((119 + ((half >> 7) & 15)) << 7) | (half & 0x7f)) << shift;
  }
  return word;
}

// Draw a fractional FP32 value in [-128, 128), using the high 24 engine bits.
inline uint32_t random_f32_bits(std::minstd_rand &random) {
  return bit_cast<uint32_t>(float(int(random() >> 7) - 0x800000) * 0x1p-16f);
}

// Draw finite fractional 16-bit inputs with magnitudes below four. Narrow
// outputs then exercise rounding without flooding the matrix with overflow.
inline uint16_t random_f16_bits(std::minstd_rand &random) {
  uint32_t raw = uint32_t(random()) >> 15;
  return uint16_t((raw & 0x8000) | ((12 + ((raw >> 10) & 3)) << 10) | (raw & 0x3ff));
}

// Draw a finite BF16 encoding with a modest exponent from the engine's high bits.
inline uint16_t random_bf16_bits(std::minstd_rand &random) {
  uint32_t raw = uint32_t(random()) >> 15;
  return uint16_t((raw & 0x8000) | ((124 + ((raw >> 7) & 3)) << 7) | (raw & 0x7f));
}

// E4M3FN has NaN but no infinity. E5M2 has both NaN and infinity.
static const uint8_t kFP8QuietNaNBits = 0x7f;
static const uint8_t kBF8QuietNaNBits = 0x7e;
static const uint8_t kBF8PositiveInfinityBits = 0x7c;

// Draw a finite FP8 encoding with a modest exponent from the engine's high bits.
inline uint8_t random_fp8_bits(std::minstd_rand &random) {
  uint32_t raw = uint32_t(random()) >> 23;
  return uint8_t((raw & 0x80) | ((5 + ((raw >> 3) & 3)) << 3) | (raw & 7));
}

// Draw a finite BF8 encoding with a modest exponent from the engine's high bits.
inline uint8_t random_bf8_bits(std::minstd_rand &random) {
  uint32_t raw = uint32_t(random()) >> 23;
  return uint8_t((raw & 0x80) | ((13 + ((raw >> 2) & 3)) << 2) | (raw & 3));
}

// IEEE FP16-to-FP32 widening is exact; no GPU arithmetic is modeled here.
inline float f16_value(uint16_t bits) {
  int exponent = (bits >> 10) & 31;
  uint32_t fraction = bits & 0x3ff;
  if (exponent == 31) {
    return bit_cast<float>((uint32_t(bits & 0x8000) << 16) | 0x7f800000U | (fraction << 13));
  }
  float magnitude = exponent ? std::ldexp(float(1024 + fraction), exponent - 25)
                             : std::ldexp(float(fraction), -24);
  return bits & 0x8000 ? -magnitude : magnitude;
}

// Widen BF16 bits to FP32 without changing the encoded sign or payload.
inline float bf16_value(uint16_t bits) { return bit_cast<float>(uint32_t(bits) << 16); }

// Match NaNs by classification. Loose semantics compare signed infinities
// exactly and finite values with atol/rtol; other semantics compare bits.
// The caller must expect exact execution for fallback/compatibility modes.
inline void expect_f32(uint32_t actual, float expected, uint64_t semantics, double atol,
                       double rtol) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(bit_cast<float>(actual)));
  } else if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
    EXPECT_EQ(actual, bit_cast<uint32_t>(expected));
  } else if (std::isinf(expected)) {
    EXPECT_EQ(bit_cast<float>(actual), expected);
  } else {
    EXPECT_LE(std::abs(double(bit_cast<float>(actual)) - expected),
              atol + rtol * std::abs(expected));
  }
}

} // namespace rocjitsu::arith::test
