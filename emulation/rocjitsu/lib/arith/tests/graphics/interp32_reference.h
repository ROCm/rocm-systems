// SPDX-License-Identifier: MIT

// FP32 interpolation with quad-lane source selection.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Reinterpret FP32 bits as a float without numeric conversion.
inline float interp32_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

// Return the raw FP32 representation without numeric conversion.
inline uint32_t interp32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}

// Translate a compact test scenario into FP32 interpolation modifiers.
inline uint32_t interp32_mode(unsigned m, unsigned wait = 0) {
  return (m & 7) | (m & 8 ? RJ_ARITH_ALU_CLAMP : 0) | (wait << RJ_ARITH_INTERP_WAIT_EXP_SHIFT);
}

// Compute FP32 interpolation from its quad-selected sources and per-lane inputs.
inline float interp32_reference(bool p2, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                int lane, unsigned m) {
  unsigned quad = lane / 4 * 4;
  long double x = interp32_float(a[quad + (p2 ? 2 : 1)]), y = interp32_float(b[lane]),
              z = interp32_float(c[p2 ? lane : quad]);
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
