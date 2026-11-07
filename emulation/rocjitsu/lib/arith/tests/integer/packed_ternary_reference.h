// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed 16-bit multiply-add and selections with saturation.

#pragma once

#include "rj_arith/detail/rj_arith_common.h"

#include <algorithm>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class PackedTernary { Mad, Min, Max, AddMin, AddMax };

inline uint32_t packed_ternary_reference(bool sign, uint32_t a, uint32_t b, uint32_t c,
                                         uint32_t mode,
                                         PackedTernary operation = PackedTernary::Mad) {
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
    int64_t value = 0;
    const int64_t low = sign ? -32768 : 0, high = sign ? 32767 : 65535;
    switch (operation) {
    case PackedTernary::Mad:
      value = input[0] * input[1] + input[2];
      if (mode & RJ_ARITH_PK_CLAMP) {
        value = std::clamp(value, low, high);
      }
      break;
    case PackedTernary::Min:
      value = std::min({input[0], input[1], input[2]});
      break;
    case PackedTernary::Max:
      value = std::max({input[0], input[1], input[2]});
      break;
    case PackedTernary::AddMin:
    case PackedTernary::AddMax:
      value = std::clamp(input[0] + input[1], low, high);
      value = operation == PackedTernary::AddMin ? std::min(value, input[2])
                                                 : std::max(value, input[2]);
      if (sign && (mode & RJ_ARITH_PK_CLAMP)) {
        value = std::max(value, int64_t(0));
      }
      break;
    }
    if (sign && (mode & RJ_ARITH_PK_CLAMP) &&
        (operation == PackedTernary::Min || operation == PackedTernary::Max)) {
      value = std::max(value, int64_t(0));
    }
    result |= (uint32_t(value) & 65535) << (half * 16);
  }
  return result;
}

} // namespace rocjitsu::arith::test
