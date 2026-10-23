// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 interpolation with staged rounding and selected-halfword output.

#pragma once

#include "fp/mixed_fma_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Translate a compact test scenario into FP16 interpolation modifiers.
inline uint32_t interp16_mode(int op, unsigned m, unsigned wait = 0) {
  return (m & 7) | (m & 8 ? RJ_ARITH_ALU_CLAMP : 0) | (m & 16 ? RJ_ARITH_ALU_HIGH_A : 0) |
         (m & 32 ? (op & 1 ? RJ_ARITH_ALU_HIGH_D : RJ_ARITH_ALU_HIGH_C) : 0) |
         (wait << RJ_ARITH_INTERP_WAIT_EXP_SHIFT);
}

// Compute the selected interpolation stage from quad-routed inputs, preserving an unselected
// destination half.
inline uint32_t interp16_reference(int op, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                   uint32_t old, int lane, unsigned m, bool saturate) {
  unsigned quad = lane / 4 * 4;
  uint32_t mode = (m & 7) | (m & 8 ? RJ_ARITH_ALU_CLAMP : 0) | (m & 16 ? RJ_ARITH_ALU_HIGH_A : 0) |
                  RJ_ARITH_MIX_F16_A;
  if (!(op & 1)) {
    mode |= RJ_ARITH_MIX_F16_C | (m & 32 ? RJ_ARITH_ALU_HIGH_C : 0);
  }
  return mixed_fma_reference::evaluate(op & 1 ? (m & 32 ? 2 : 1) : 0, a[quad + (op & 1 ? 2 : 1)],
                                       b[lane], c[op & 1 ? lane : quad], old, mode, saturate,
                                       op >= 2);
}

} // namespace rocjitsu::arith::test
