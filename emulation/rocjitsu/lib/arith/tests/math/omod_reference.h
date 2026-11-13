// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating-point output scaling and clamping reference rules.

#pragma once

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Apply the test reference's FP32 output scaling, tininess, and clamp rules.
inline float omod_f32_reference(float value, uint32_t mode) {
  unsigned omod = (mode >> 6) & 3;
  if (!omod) {
    return value;
  }
  float magnitude = std::abs(value);
  if (magnitude < std::ldexp(1.0f, -126)) {
    return 0.0f;
  }
  if (omod == 3 && magnitude < std::ldexp(1.0f, -125)) {
    return std::copysign(0.0f, value);
  }
  return std::ldexp(value, omod == 3 ? -1 : int(omod));
}

// Apply the test reference's FP64 output scaling and clamp rules.
inline double omod_f64_reference(double value, uint32_t mode) {
  unsigned omod = (mode >> 6) & 3;
  if (!omod) {
    return value;
  }
  double magnitude = std::abs(value);
  if (magnitude < std::ldexp(1.0, -1022)) {
    return 0.0;
  }
  if (omod == 3 && magnitude < std::ldexp(1.0, -1021)) {
    return std::copysign(0.0, value);
  }
  return std::ldexp(value, omod == 3 ? -1 : int(omod));
}

} // namespace rocjitsu::arith::test
