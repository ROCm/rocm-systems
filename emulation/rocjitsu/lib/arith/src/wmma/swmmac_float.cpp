// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar loose FP32 accumulation for decoded floating-point SWMMAC operands.
// Traverses compressed K positions and applies position-dependent B negation;
// callers handle operand decoding and result packing.

#include "wmma/swmmac_float.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Accumulate decoded sparse products into input.acc using loose FP32 arithmetic.
// A/C modifiers must already be applied; mode supplies B pair-position negation.
void swmmac_float_accumulate(uint32_t mode, SwmmacFloatInputs &input) {
  const uint32_t flip[] = {mode & RJ_ARITH_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & RJ_ARITH_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      float acc = input.acc[row][col];
      for (int ck = 0; ck < 16; ++ck) {
        float bv = rocjitsu::arith::as_float(
            rocjitsu::arith::as_bits(input.b[input.selected[row][ck]][col]) ^ flip[ck & 1]);
        acc = std::fma(input.a[row][ck], bv, acc);
      }
      input.acc[row][col] = acc;
    }
  }
}

} // namespace rocjitsu::arith
