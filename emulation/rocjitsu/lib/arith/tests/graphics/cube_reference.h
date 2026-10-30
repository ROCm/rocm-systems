// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cube-map coordinates, face selection, and major-axis magnitude.

#pragma once

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Numeric reference using ordered FP comparisons and double-precision scaling.
inline uint32_t cube_reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  uint32_t raw[] = {a, b, c};
  float compare[3];
  for (int i = 0; i < 3; ++i) {
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      raw[i] &= 0x7fffffff;
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      raw[i] ^= 0x80000000;
    }
    compare[i] = std::bit_cast<float>(raw[i]);
    if (std::fabs(compare[i]) < 0x1p-126f) {
      compare[i] = 0;
    }
  }
  unsigned axis = 0;
  if (std::fabs(compare[2]) >= std::fabs(compare[0]) &&
      std::fabs(compare[2]) >= std::fabs(compare[1])) {
    axis = 2;
  } else if (std::fabs(compare[1]) >= std::fabs(compare[0])) {
    axis = 1;
  }
  bool negative = compare[axis] < 0;
  uint32_t result;
  if (op == 0) {
    result = std::bit_cast<uint32_t>(float(2 * axis + negative));
  } else if (op == 1) {
    int index = axis == 0 ? 2 : 0;
    bool flip = axis == 0 ? !negative : axis == 2 && negative;
    result = raw[index] ^ (flip ? 0x80000000 : 0);
  } else if (op == 2) {
    result = axis == 1 ? raw[2] ^ (negative ? 0x80000000 : 0) : raw[1] ^ 0x80000000;
  } else {
    if (std::isnan(compare[axis]) || std::isinf(compare[axis])) {
      result = raw[axis];
    } else if (compare[axis] == 0) {
      result = 0;
    } else {
      double value = 2.0 * compare[axis];
      result = std::fabs(value) > 0x1.fffffep127 ? ((raw[axis] & 0x80000000) | 0x7f800000)
                                                 : std::bit_cast<uint32_t>(float(value));
    }
  }
  if (std::isnan(std::bit_cast<float>(result))) {
    result |= 0x00400000;
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod && std::isfinite(std::bit_cast<float>(result))) {
    float input = std::bit_cast<float>(result);
    const double scales[] = {1, 2, 4, 0.5};
    double output = double(input) * scales[omod];
    if (std::fabs(input) < 0x1p-126f) {
      result = 0;
    } else if (std::fabs(output) < 0x1p-126) {
      result &= 0x80000000;
    } else if (std::fabs(output) > 0x1.fffffep127) {
      result = (result & 0x80000000) | 0x7f800000;
    } else {
      result = std::bit_cast<uint32_t>(float(output));
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    float value = std::bit_cast<float>(result);
    if (!(value > 0)) {
      result = 0;
    } else if (value > 1) {
      result = 0x3f800000;
    }
  }
  return result;
}

} // namespace rocjitsu::arith::test
