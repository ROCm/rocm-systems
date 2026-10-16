// SPDX-License-Identifier: MIT

// Shared unary math test scenarios and reference selection.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "common/floating_point.h"
#include "math/omod_reference.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent higher-precision reference, with explicit ties-to-even and
// binary32 rounding before output scaling.
inline float unary_reference(int op, float input, uint32_t flags) {
  double x = input;
  if (flags & RJ_ARITH_ALU_ABS_A) {
    x = std::abs(x);
  }
  if (flags & RJ_ARITH_ALU_NEG_A) {
    x = -x;
  }
  bool flush = op >= 4 && op <= 8;
  if (flush && std::abs(x) < std::ldexp(1.0, -126)) {
    x = std::copysign(0.0, x);
  }
  double y = 0;
  switch (op) {
  case 10: {
    int exponent;
    y = std::isfinite(x) ? std::frexp(x, &exponent) : x;
    break;
  }
  case 0:
    y = std::trunc(x);
    break;
  case 1:
    y = std::ceil(x);
    break;
  case 2:
    y = x;
    if (std::isfinite(x) && x != 0) {
      double lo = std::floor(x), fraction = x - lo;
      y = lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2) != 0));
      y = std::copysign(std::abs(y), x);
    }
    break;
  case 3:
    y = std::floor(x);
    break;
  case 4:
    y = std::sqrt(x);
    break;
  case 5:
    y = 1 / x;
    break;
  case 6:
    y = 1 / std::sqrt(x);
    break;
  case 7:
    y = std::exp2(x);
    break;
  case 9:
    y = x - std::floor(x);
    if (y > double(bit_cast<float>(0x3f7fffff))) {
      y = bit_cast<float>(0x3f7fffff);
    }
    break;
  case 8:
    y = std::log2(x);
    break;
  }
  float result = float(y);
  if (flush && std::abs(result) < std::ldexp(1.0f, -126)) {
    result = std::copysign(0.0f, result);
  }
  result = rocjitsu::arith::test::omod_f32_reference(result, flags);
  if (flags & RJ_ARITH_ALU_CLAMP) {
    result = std::isnan(result) || result <= 0 ? 0 : std::min(result, 1.0f);
  }
  return result;
}

} // namespace rocjitsu::arith::test
