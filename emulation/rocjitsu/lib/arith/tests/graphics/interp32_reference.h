// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 interpolation with quad-lane source selection.

#pragma once

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Translate a compact test scenario into FP32 interpolation modifiers.
inline uint32_t interp32_mode(unsigned m, unsigned wait = 0) {
  return (m & 7) | (m & 8 ? RJ_ARITH_ALU_CLAMP : 0) | (wait << RJ_ARITH_INTERP_WAIT_EXP_SHIFT);
}

// Compute FP32 interpolation from its quad-selected sources and per-lane inputs.
inline float interp32_reference(bool p2, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                int lane, unsigned m) {
  unsigned quad = lane / 4 * 4;
  long double x = std::bit_cast<float>(a[quad + (p2 ? 2 : 1)]), y = std::bit_cast<float>(b[lane]),
              z = std::bit_cast<float>(c[p2 ? lane : quad]);
  if (m & 1) {
    x = -x;
  }
  if (m & 2) {
    y = -y;
  }
  if (m & 4) {
    z = -z;
  }
  float value = float(std::fma(x, y, z));
  if (m & 8) {
    value = !(value > 0) ? 0 : value > 1 ? 1 : value;
  }
  return value;
}

} // namespace rocjitsu::arith::test
