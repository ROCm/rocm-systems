// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test-side FP encoding, operand access, modifiers, and numeric comparisons.

#pragma once

#include "common/floating_point.h"
#include "common/half_reference.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Read/write an element across independent VGPR pointers. FP16 writes preserve
// the other half; FP64 uses low/high words in successive VGPRs.
inline uint64_t read_fp(const uint32_t *const *v, int lane, int width, int half = 0) {
  if (width == 16) {
    return uint16_t(v[0][lane] >> (16 * half));
  }
  if (width == 64) {
    return v[0][lane] | (uint64_t(v[1][lane]) << 32);
  }
  return v[0][lane];
}

// Write an FP16/FP32/FP64 encoding into the corresponding VGPR words or halfword.
inline void write_fp(uint32_t *const *v, int lane, int width, uint64_t bits, int half = 0) {
  if (width == 16) {
    v[0][lane] = (v[0][lane] & ~(0xffffU << (16 * half))) | (uint32_t(bits) << (16 * half));
  } else {
    v[0][lane] = uint32_t(bits);
    if (width == 64) {
      v[1][lane] = uint32_t(bits >> 32);
    }
  }
}

// Decode raw FP16/FP32/FP64 bits as a double.
inline double fp_value(uint64_t bits, int width) {
  return width == 16   ? half_value(uint16_t(bits))
         : width == 32 ? double(std::bit_cast<float>(uint32_t(bits)))
                       : std::bit_cast<double>(bits);
}

// Encode a double in the requested floating-point width.
inline uint64_t fp_bits(double value, int width) {
  return width == 16   ? half_bits(value)
         : width == 32 ? uint64_t(std::bit_cast<uint32_t>(float(value)))
                       : std::bit_cast<uint64_t>(value);
}

// Use high engine bits, with fractional significands and modest exponents.
// FP64 draws a full 52-bit fraction from two independent engine results.
inline uint64_t random_fp(std::minstd_rand &random, int width) {
  if (width == 16) {
    return random_f16_bits(random);
  }
  if (width == 32) {
    return random_f32_bits(random);
  }
  uint64_t high = uint64_t(random()) >> 4, low = uint64_t(random()) >> 5;
  return ((high & (1ULL << 26)) << 37) | ((1020ULL + ((high >> 23) & 7)) << 52) |
         ((high & 0x3ffffffULL) << 26) | low;
}

// Compare numeric results, leaving representation checks for untouched lanes
// and half-registers to the caller. Exact modes compare bits except NaN payloads.
inline void expect_fp(uint64_t actual, uint64_t expected, int width, uint64_t semantics,
                      double atol, double rtol) {
  double value = fp_value(expected, width), result = fp_value(actual, width);
  if (std::isnan(value)) {
    EXPECT_TRUE(std::isnan(result));
  } else if (semantics != RJ_ARITH_SEMANTICS_LOOSE) {
    EXPECT_EQ(actual, expected);
  } else if (std::isinf(value)) {
    EXPECT_EQ(result, value);
  } else {
    EXPECT_LE(std::abs(result - value), atol + rtol * std::abs(value));
  }
}

// IEEE selection with signed-zero ordering. These test references cover quiet
// NaNs. Directed guest-control cases live in v_min_max_test.cpp; loose results
// do not promise a particular NaN payload or signed-zero choice.
inline double select_fp(double x, double y, bool maximum, bool propagate = false) {
  if (std::isnan(x) || std::isnan(y)) {
    return propagate ? std::numeric_limits<double>::quiet_NaN() : std::isnan(x) ? y : x;
  }
  if (x == 0 && y == 0) {
    return (maximum ? std::signbit(x) && std::signbit(y) : std::signbit(x) || std::signbit(y))
               ? -0.0
               : 0.0;
  }
  return maximum ? std::max(x, y) : std::min(x, y);
}

// Return the test reference's three-value median.
inline double median_fp(double x, double y, double z) {
  if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
    return select_fp(select_fp(x, y, false), z, false);
  }
  double maximum = std::max({x, y, z});
  return maximum == x   ? select_fp(y, z, true)
         : maximum == y ? select_fp(x, z, true)
                        : select_fp(x, y, true);
}

// Apply source sign modifiers; selection of packed halves stays in each test.
inline double modified_fp(double value, uint64_t mode, int operand) {
  if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
    value = std::abs(value);
  }
  if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
    value = -value;
  }
  return value;
}

// Universal early-rejection contract, without an inventory of entry points or
// their modifier capabilities. The caller checks all operand/status storage.
template <typename Call> void expect_invalid_fp_requests(Call call) {
  for (int semantics = 4; semantics <= 6; ++semantics) {
    EXPECT_EQ(call(uint64_t(semantics) << 16, 0), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  EXPECT_EQ(call(1ULL << 63, 0), RJ_ARITH_ERROR_INVALID_FLAGS);
  // Unsupported reporting may be rejected before other flags are examined.
  int error = call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | (1ULL << 63), 0);
  EXPECT_TRUE(error == RJ_ARITH_ERROR_INVALID_FLAGS ||
              error == RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS)
      << error;
  EXPECT_EQ(call(0, 1ULL << 63), RJ_ARITH_ERROR_INVALID_FLAGS);
}

} // namespace rocjitsu::arith::test
