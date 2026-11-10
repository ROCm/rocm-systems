// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed 16-bit multiply-add and saturation.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
inline uint32_t packed_mad_reference(bool sign, uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  uint32_t result = 0;
  const uint32_t words[] = {a, b, c};
  for (int half = 0; half < 2; ++half) {
    int64_t input[3];
    for (int source = 0; source < 3; ++source) {
      bool high = half ? !(mode & (RJ_ARITH_PK_HI_A_LOW << source))
                       : bool(mode & (RJ_ARITH_PK_LO_A_HIGH << source));
      input[source] = (words[source] >> (high ? 16 : 0)) & 65535;
      if (sign && input[source] > 32767) {
        input[source] -= 65536;
      }
    }
    int64_t value = input[0] * input[1] + input[2];
    if (mode & RJ_ARITH_PK_CLAMP) {
      const int64_t low = sign ? -32768 : 0, high = sign ? 32767 : 65535;
      if (value < low) {
        value = low;
      }
      if (value > high) {
        value = high;
      }
    }
    result |= (uint32_t(value) & 65535) << (half * 16);
  }
  return result;
}

} // namespace rocjitsu::arith::test
