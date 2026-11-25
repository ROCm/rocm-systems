// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited gfx942 ordinary FP16 arithmetic-step exceptions. Output modifiers,
// output flushing and finite saturation do not alter this legacy classifier.
#pragma once

#include "common/bits.h"
#include "common/float_encoding.h"
#include "fp/binary.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Classify selected, modified operands before any aliased destination is written.
// Finite half addition/multiplication is exact in double under the default host environment.
template <Binary Op>
uint32_t gfx942_half_causes(uint64_t flags, uint64_t exec_mask, uint32_t modifiers,
                            const uint32_t *a, const uint32_t *b) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto source = [&](uint32_t raw, bool high, unsigned index) {
    uint16_t bits = uint16_t(raw >> (high ? 16 : 0));
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fff) < 0x400) {
      bits &= 0x8000;
    }
    if (modifiers & (RJ_ARITH_ALU_ABS_A << index)) {
      bits &= 0x7fff;
    }
    if (modifiers & (RJ_ARITH_ALU_NEG_A << index)) {
      bits ^= 0x8000;
    }
    return bits;
  };
  uint32_t causes = 0;
  for (int lane = 0; exec_mask; ++lane, exec_mask >>= 1) {
    if (!(exec_mask & 1)) {
      continue;
    }
    uint16_t a_bits = source(a[lane], modifiers & RJ_ARITH_ALU_HIGH_A, 0);
    uint16_t b_bits = source(b[lane], modifiers & RJ_ARITH_ALU_HIGH_B, 1);
    if constexpr (Op == Binary::Sub) {
      b_bits ^= 0x8000;
    }
    if constexpr (Op == Binary::Subrev) {
      a_bits ^= 0x8000;
    }
    const unsigned x = a_bits & 0x7fff, y = b_bits & 0x7fff;
    if (x >= 0x7c00 || y >= 0x7c00) {
      const bool snan = (x > 0x7c00 && !(a_bits & 0x200)) || (y > 0x7c00 && !(b_bits & 0x200));
      const bool invalid = Op == Binary::Mul
                               ? (x == 0 && y == 0x7c00) || (y == 0 && x == 0x7c00)
                               : x == 0x7c00 && y == 0x7c00 && ((a_bits ^ b_bits) & 0x8000);
      if (snan || invalid) {
        causes |= RJ_ARITH_EXCEPTION_INVALID;
      }
      continue;
    }
    if ((x && x < 0x400) || (y && y < 0x400)) {
      causes |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    }
    const double lhs = f16_to_float(a_bits), rhs = f16_to_float(b_bits);
    const double exact = Op == Binary::Mul ? lhs * rhs : lhs + rhs;
    const double magnitude = __builtin_fabs(exact);
    const bool away = (rounding == 1 && exact > 0) || (rounding == 2 && exact < 0);
    // Preserve the old unbounded-exponent overflow test, independently of FP16_OVFL.
    if (magnitude >= 65536.0 || (rounding == 0 && magnitude >= 65520.0) ||
        (away && magnitude > 65504.0)) {
      causes |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
      continue;
    }
    const uint16_t rounded = half_fma_narrow(bit_cast<uint64_t>(exact), false, rounding);
    if (exact != f16_to_float(rounded)) {
      causes |= RJ_ARITH_EXCEPTION_INEXACT;
      if ((rounded & 0x7fff) < 0x400) {
        causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
    }
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
