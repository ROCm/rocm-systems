// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deterministic floating-point inputs, bit representations, and exact/loose comparisons.

#pragma once

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
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
  return std::bit_cast<uint32_t>(float(int(random() >> 7) - 0x800000) * 0x1p-16f);
}

// Draw finite fractional 16-bit inputs with magnitudes below four. Narrow
// outputs then exercise rounding without flooding the matrix with overflow.
inline uint16_t random_f16_bits(std::minstd_rand &random) {
  uint32_t raw = uint32_t(random()) >> 15;
  return uint16_t((raw & 0x8000) | ((12 + ((raw >> 10) & 3)) << 10) | (raw & 0x3ff));
}

// Match NaNs by classification. Loose semantics compare signed infinities
// exactly and finite values with atol/rtol; other semantics compare bits.
// The caller must expect exact execution for fallback/compatibility modes.
inline void expect_f32(uint32_t actual, float expected, uint64_t semantics, double atol,
                       double rtol) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(actual)));
  } else if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
    EXPECT_EQ(actual, std::bit_cast<uint32_t>(expected));
  } else if (std::isinf(expected)) {
    EXPECT_EQ(std::bit_cast<float>(actual), expected);
  } else {
    EXPECT_LE(std::abs(double(std::bit_cast<float>(actual)) - expected),
              atol + rtol * std::abs(expected));
  }
}

} // namespace rocjitsu::arith::test
