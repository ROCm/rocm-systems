// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lit-multiply special values and gfx11/gfx12 modifier behavior.

#pragma once

#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Evaluate the lighting rule with separate source and output modifiers.
inline uint32_t mullit_reference(uint32_t a, uint32_t b, uint32_t c, uint32_t mode,
                                 bool rdna3 = false, uint64_t flags = 0) {
  const uint32_t words[] = {a, b, c};
  float input[3];
  for (int i = 0; i < 3; ++i) {
    uint32_t word = words[i];
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (word & 0x7fffffff) < 0x00800000) {
      word &= 0x80000000;
    }
    input[i] = std::bit_cast<float>(word);
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      input[i] = std::fabs(input[i]);
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      input[i] = -input[i];
    }
  }
  float value;
  if (std::isnan(input[1]) || input[1] <= -std::bit_cast<float>(0x7f7fffff) ||
      std::isnan(input[2]) || input[2] <= 0) {
    value = -std::bit_cast<float>(0x7f7fffff);
  } else if (input[0] == 0 || input[1] == 0) {
    value = 0;
  } else {
    value = input[0] * input[1];
  }
  unsigned scale = (mode >> 6) & 3;
  if (rdna3 && (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE))) {
    scale = 0;
  }
  if (scale) {
    if (!rdna3 && std::fabs(value) < std::bit_cast<float>(0x00800000)) {
      value = 0;
    }
    value = std::ldexp(value, scale == 3 ? -1 : int(scale));
    if (std::fabs(value) < std::bit_cast<float>(0x00800000)) {
      value = rdna3 ? 0.0f : std::copysign(0.0f, value);
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    if (std::isnan(value) || value <= 0) {
      value = 0;
    } else if (value > 1) {
      value = 1;
    }
  }
  if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
      std::fabs(value) < std::bit_cast<float>(0x00800000)) {
    value = std::copysign(0.0f, value);
  }
  uint32_t word = std::bit_cast<uint32_t>(value);
  return (word & 0x7fffffff) > 0x7f800000 ? 0x7fc00000 : word;
}

} // namespace rocjitsu::arith::test
