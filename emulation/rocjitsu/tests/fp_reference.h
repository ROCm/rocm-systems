// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Small standard floating-point references for decoded arithmetic tests.
// These use the test process's default host FP environment.
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace rocjitsu::test {

struct FloatingOutput {
  // Raw OMOD: 0 none, 1 multiply by two, 2 multiply by four, 3 halve.
  unsigned omod;
  // Guest rounding: 0 nearest-even, 1 up, 2 down, 3 toward zero.
  unsigned rounding;
  bool clamp;
  bool nan_to_zero;
};

// Clamp to [+0, 1], preserving NaNs or replacing them with +0 as requested.
template <typename Float> Float clamp_floating(Float value, bool nan_to_zero) {
  static_assert(std::is_floating_point_v<Float>);
  if (std::isnan(value)) {
    return nan_to_zero ? Float(0) : value;
  }
  return value <= 0 ? Float(0) : value > 1 ? Float(1) : value;
}

// Scale by OMOD and clamp in the destination format. OMOD flushes initially tiny
// values to +0 and newly tiny results to signed zero; overflow follows guest rounding.
template <typename Float> Float finish_floating(Float value, FloatingOutput output) {
  if (std::isnan(value)) {
    return output.clamp && output.nan_to_zero ? Float(0) : value;
  }
  if (output.omod) {
    if (std::abs(value) < std::numeric_limits<Float>::min()) {
      value = Float(0);
    } else if (std::isfinite(value)) {
      const Float scaled = std::scalbn(value, output.omod == 3 ? -1 : int(output.omod));
      const bool infinity = output.rounding == 0 || output.rounding == (value < 0 ? 2U : 1U);
      value =
          std::isinf(scaled) && !infinity ? std::copysign(std::numeric_limits<Float>::max(), value)
          : std::abs(scaled) < std::numeric_limits<Float>::min() ? std::copysign(Float(0), value)
                                                                 : scaled;
    }
  }
  if (output.clamp) {
    return rocjitsu::test::clamp_floating(value, output.nan_to_zero);
  }
  return value;
}

} // namespace rocjitsu::test
