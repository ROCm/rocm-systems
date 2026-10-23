// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed DOT4/DOT8 source-half selection shared by scalar and SIMD dispatch.

#pragma once

#include "rj_arith/rj_arith.h"
#include <stdint.h>

namespace rocjitsu::arith {

inline constexpr uint32_t kDotHalfSelection =
    RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_LO_B_HIGH | RJ_ARITH_DOT_HI_A_LOW | RJ_ARITH_DOT_HI_B_LOW;

// Snapshot selected A/B halves before invoking a kernel; C and destination retain their
// original storage. The caller handles the default layout without staging.
template <typename Call>
int with_dot_halves(uint32_t modifiers, const uint32_t *const *a, const uint32_t *const *b,
                    Call call) {
  alignas(64) uint32_t selected[2][32];
  for (int operand = 0; operand < 2; ++operand) {
    const auto *source = operand ? b[0] : a[0];
    int low = modifiers & (RJ_ARITH_DOT_LO_A_HIGH << operand) ? 16 : 0;
    int high = modifiers & (RJ_ARITH_DOT_HI_A_LOW << operand) ? 0 : 16;
    for (int lane = 0; lane < 32; ++lane) {
      selected[operand][lane] = ((source[lane] >> low) & 0xffffU) | ((source[lane] >> high) << 16);
    }
  }
  const uint32_t *av = selected[0], *bv = selected[1];
  return call(modifiers & ~kDotHalfSelection, &av, &bv);
}

} // namespace rocjitsu::arith
