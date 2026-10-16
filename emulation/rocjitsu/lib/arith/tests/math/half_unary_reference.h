// SPDX-License-Identifier: MIT

// FP16 unary arithmetic and output-modifier reference rules.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "common/half_reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Evaluate the selected loose unary operation on a decoded value.
inline double half_unary(int op, double x) {
  switch (op) {
  case 0:
    return std::trunc(x);
  case 1:
    return std::ceil(x);
  case 2: {
    if (!std::isfinite(x) || x == 0) {
      return x;
    }
    double low = std::floor(x), tail = x - low;
    double result = low + (tail > 0.5 || (tail == 0.5 && std::fmod(low, 2) != 0));
    return std::copysign(std::abs(result), x);
  }
  case 3:
    return std::floor(x);
  case 4:
    return std::sqrt(x);
  case 5:
    return 1 / x;
  case 6:
    return 1 / std::sqrt(x);
  case 7: {
    double result = std::exp2(x);
    // Preserve finite-overflow provenance through subsequent output scaling.
    return std::isfinite(x) && result > 1e100 ? 1e100 : result;
  }
  case 8:
    return std::log2(x);
  case 9: {
    double result = x - std::floor(x);
    return result > 1.0 - 1.0 / 2048 ? 1.0 - 1.0 / 2048 : result;
  }
  default: {
    int exponent;
    return std::isfinite(x) ? std::frexp(x, &exponent) : x;
  }
  }
}

// Compute the FP16 unary reference with source selection, guest controls, and output modifiers.
inline uint16_t half_unary_reference(int op, uint32_t input, uint32_t mode, bool saturate,
                                     uint64_t fp_flags = 0) {
  double x =
      rocjitsu::arith::test::half_value(uint16_t(input >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)));
  if ((op == 7 || op == 8) && (fp_flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
      std::abs(x) < 0x1p-14) {
    x = std::copysign(0.0, x);
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    x = std::abs(x);
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    x = -x;
  }
  double result = rocjitsu::arith::test::half_unary(op, x);
  if (op >= 4 && op <= 8) {
    if ((op == 5 || op == 6) && x == 0 && saturate) {
      result = std::copysign(65504.0, result);
    }
    if (op == 8 && x == 0 && saturate) {
      result = -65504;
    }
    result = rocjitsu::arith::test::half_value(rocjitsu::arith::test::half_bits(result, saturate));
  }
  if ((op == 7 || op == 8) && (fp_flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
      std::abs(result) < 0x1p-14) {
    result = std::copysign(0.0, result);
  }
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & RJ_ARITH_ALU_OMOD_HALF) {
    if (std::abs(result) < 0x1p-14) {
      result = 0;
    } else if ((mode & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF &&
               std::abs(result) < 0x1p-13) {
      result = std::copysign(0.0, result);
    }
  }
  result *= scales[(mode >> 6) & 3];
  if (mode & RJ_ARITH_ALU_CLAMP) {
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  }
  return rocjitsu::arith::test::half_bits(result, saturate);
}

// Map a compact scenario index to FP16 source/output modifiers.
inline uint32_t half_unary_modifiers(int variant) {
  return (variant & 1 ? RJ_ARITH_ALU_NEG_A : 0) | (variant & 2 ? RJ_ARITH_ALU_ABS_A : 0) |
         ((variant & 28) << 4) | (variant & 32 ? RJ_ARITH_ALU_HIGH_A : 0) |
         (variant & 64 ? RJ_ARITH_ALU_HIGH_D : 0);
}

} // namespace rocjitsu::arith::test
