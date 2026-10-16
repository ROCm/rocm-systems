// SPDX-License-Identifier: MIT

// Integer narrowing, extension, and packed halfword conversion.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent reference inherited from tests/conversion/integer_conversion_reference.h.
// Retain its rounding and special-value model; do not call production helpers.

// Compute the selected integer narrowing/extension operation from raw source words.
inline uint32_t integer_conversion_reference(int op, uint32_t a, uint32_t b, uint32_t mode) {
  if (op < 2) {
    unsigned half = (a >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
    int64_t value = half;
    if (op == 0 && half >= 32768) {
      value -= 65536;
    }
    return uint32_t(value);
  }
  uint32_t result = 0;
  for (int reg = 0; reg < 2; ++reg) {
    uint32_t raw = reg ? b : a;
    int64_t value = raw;
    if (op == 2 && raw >= 0x80000000) {
      value -= 4294967296LL;
    }
    int64_t low = op == 2 ? -32768 : 0, high = op == 2 ? 32767 : 65535;
    value = value < low ? low : value > high ? high : value;
    result |= uint32_t(uint16_t(value)) << (16 * reg);
  }
  return result;
}

} // namespace rocjitsu::arith::test
