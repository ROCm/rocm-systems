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
// the other half.
inline uint64_t read_fp(const uint32_t *const *v, int lane, int width, int half = 0) {
  if (width == 16) {
    return uint16_t(v[0][lane] >> (16 * half));
  }
  return v[0][lane];
}

// Write an FP16/FP32 encoding into the corresponding VGPR words or halfword.
inline void write_fp(uint32_t *const *v, int lane, int width, uint64_t bits, int half = 0) {
  if (width == 16) {
    v[0][lane] = (v[0][lane] & ~(0xffffU << (16 * half))) | (uint32_t(bits) << (16 * half));
  } else {
    v[0][lane] = uint32_t(bits);
  }
}

// Decode raw FP16/FP32 bits as a double.
inline double fp_value(uint64_t bits, int width) {
  return width == 16 ? half_value(uint16_t(bits)) : double(std::bit_cast<float>(uint32_t(bits)));
}

// Encode a double in the requested floating-point width.
inline uint64_t fp_bits(double value, int width) {
  return width == 16 ? half_bits(value) : uint64_t(std::bit_cast<uint32_t>(float(value)));
}

// Use high engine bits, with fractional significands and modest exponents.
inline uint64_t random_fp(std::minstd_rand &random, int width) {
  if (width == 16) {
    return random_f16_bits(random);
  }
  return random_f32_bits(random);
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

// Apply source sign modifiers; selection of packed halves stays in each test.
inline double modified_fp(double value, uint64_t instruction_flags, int operand) {
  if (instruction_flags & (RJ_ARITH_ALU_ABS_A << operand)) {
    value = std::abs(value);
  }
  if (instruction_flags & (RJ_ARITH_ALU_NEG_A << operand)) {
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
  EXPECT_EQ(call(0, 1ULL << 63), RJ_ARITH_ERROR_INVALID_FLAGS);
}

} // namespace rocjitsu::arith::test
