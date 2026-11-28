// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Raw-bit OMOD scaling retained by gfx10 interpolation. Numerical ALU
// instructions and their output modifiers are implemented by rj-arith.
#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu::output_modifier {

// Scale by 1, 2, 4 or 1/2. OMOD flushes tiny inputs to +0, preserves nonfinite
// encodings, and applies guest rounding when multiplication overflows.
inline uint32_t scale_f32(uint32_t bits, uint32_t omod, uint32_t round_mode) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (!omod || magnitude >= 0x7f800000u) {
    return bits;
  }
  if (magnitude < 0x00800000u) {
    return 0;
  }
  const uint32_t exponent = magnitude >> 23;
  if (omod == 3) {
    return exponent == 1 ? bits & 0x80000000u : bits - 0x00800000u;
  }
  if (exponent >= 255 - omod) {
    const bool negative = (bits >> 31) != 0;
    const bool infinity = round_mode == 0 || round_mode == (negative ? 2u : 1u);
    return (bits & 0x80000000u) | (infinity ? 0x7f800000u : 0x7f7fffffu);
  }
  return bits + (omod << 23);
}

} // namespace rocjitsu::amdgpu::output_modifier
