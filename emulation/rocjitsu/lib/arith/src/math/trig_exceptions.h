// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest sine/cosine exception classification, separate from numeric evaluation.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::trig {

// Return SIN/COS exception bits for preserved denormals and nearest-even.
// Input and result are IEEE encodings, before output modifiers. Signs do not
// affect flags. RX 9070 treats subnormal FP32 COS inputs as an exact zero input;
// FP16 COS still reports input-denormal/inexact for its subnormal inputs.
template <int Width>
inline uint32_t exceptions(uint32_t input, uint32_t result, bool cosine, uint32_t mode) {
  constexpr unsigned fraction = Width == 16 ? 10 : 23, bias = Width == 16 ? 15 : 127;
  constexpr uint32_t sign = 1U << (Width - 1), infinity = (2 * bias + 1) << fraction;
  input &= sign - 1;
  if (mode & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }
  if (input >= infinity) {
    return input == infinity || !(input & (1U << (fraction - 1))) ? RJ_ARITH_EXCEPTION_INVALID : 0;
  }
  if (!input || (Width == 32 && cosine && input < (1U << fraction))) {
    return 0;
  }
  uint32_t flags = input < (1U << fraction) ? RJ_ARITH_EXCEPTION_INPUT_DENORM : 0;
  if (mode & RJ_ARITH_ALU_OMOD_HALF) {
    return flags;
  }

  // Multiples of a quarter turn produce exact 0 or +/-1. Every other finite
  // input reports inexact, even if the rounded approximation is 0 or +/-1.
  int shift = int(fraction + bias - 2) - int(input >> fraction);
  if (shift <= 0 || (shift <= int(fraction) && !(input & ((1U << shift) - 1)))) {
    return flags;
  }
  flags |= RJ_ARITH_EXCEPTION_INEXACT;
  if (!cosine && (result & (sign - 1)) < (1U << fraction)) {
    flags |= RJ_ARITH_EXCEPTION_UNDERFLOW;
  }
  return flags;
}

} // namespace rocjitsu::arith::trig
