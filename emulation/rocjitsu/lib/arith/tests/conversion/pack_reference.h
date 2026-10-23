// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturating byte packing and selected FP16 halfword packing.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independently classify signed halves and manipulate individual FP16 fields.
inline uint32_t pack_reference(int variant, uint32_t a, uint32_t b, uint32_t d,
                               bool quiet_nan = true) {
  if (variant < 2) {
    uint32_t packed = 0;
    for (int h = 0; h < 2; ++h) {
      unsigned x = (a >> (16 * h)) & 65535;
      int byte = (x & 32768) ? 0 : (x > 255 ? 255 : x);
      packed |= byte << (8 * h);
    }
    return variant ? (d & 65535) | (packed << 16) : (d & 0xffff0000) | packed;
  }
  unsigned m = variant - 2;
  uint32_t out = 0;
  for (int h = 0; h < 2; ++h) {
    unsigned x = ((h ? b : a) >> ((m & (16u << h)) ? 16 : 0)) & 65535;
    unsigned sign = x >> 15, exponent = (x >> 10) & 31, fraction = x & 1023;
    if (quiet_nan && exponent == 31 && fraction) {
      fraction |= 512;
    }
    if (m & (1u << h)) {
      sign = 0;
    }
    if (m & (4u << h)) {
      sign ^= 1;
    }
    out |= (sign * 32768 + exponent * 1024 + fraction) << (16 * h);
  }
  return out;
}

} // namespace rocjitsu::arith::test
