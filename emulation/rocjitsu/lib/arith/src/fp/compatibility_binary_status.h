// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited FP32 binary exception policies, separate from strict-mode qualification.
// MUL shares the hardware classifier with explicit reporting.
// Uses the default host FP environment without changing its controls.

#pragma once

#include "common/alu.h"
#include "common/validation.h"
#include "fp/binary.h"
#include "fp/fma_exceptions.h"
#include "fp/mul_exceptions.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Classify one unmodified gfx12 source before guest input flushing.
inline uint32_t binary_input_causes_f32(uint32_t bits, uint64_t flags) {
  const uint32_t magnitude = bits & 0x7fffffff;
  if (magnitude > 0x7f800000 && !(magnitude & 0x400000)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  return !(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude && magnitude < 0x800000
             ? RJ_ARITH_EXCEPTION_INPUT_DENORM
             : 0;
}

// Retain rdna4/binary_fp_status.h's input, invalid-operation and pre-OMOD overflow causes
// for ADD/SUB/SUBREV and the selection operations.
template <Binary Op>
inline uint32_t gfx12_binary_causes(uint64_t flags, uint32_t modifiers, uint32_t a, uint32_t b) {
  uint32_t causes = binary_input_causes_f32(a, flags) | binary_input_causes_f32(b, flags);
  if constexpr (Op != Binary::Add && Op != Binary::Sub && Op != Binary::Subrev) {
    return causes;
  }
  a = alu_input_bits(a, 0, flags);
  b = alu_input_bits(b, 0, flags);
  const uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff;
  a = alu_input_bits(a, modifiers);
  b = alu_input_bits(b, modifiers >> 1);
  if constexpr (Op == Binary::Sub) {
    b ^= 0x80000000;
  }
  if constexpr (Op == Binary::Subrev) {
    a ^= 0x80000000;
  }
  if ((a ^ b) & 0x80000000) {
    return causes | (ma == 0x7f800000 && mb == 0x7f800000 ? RJ_ARITH_EXCEPTION_INVALID : 0);
  }
  if (ma >= 0x7f800000 || mb >= 0x7f800000 || (ma > mb ? ma : mb) < 0x7f000000) {
    return causes;
  }
  // The legacy model retains the addition residual when choosing the side of
  // an overflow boundary. FP32 operands cannot overflow or underflow FP64.
  const double x = double(as_float(a)), y = double(as_float(b)), sum = x + y;
  const double part = sum - x, residual = (x - (sum - part)) + (y - part);
  uint64_t bits = double_bits(sum);
  if (residual != 0 && !(bits & 1)) {
    bits = (sum > 0) == (residual > 0) ? bits + 1 : bits - 1;
  }
  const double exact = as_double(bits), magnitude = __builtin_fabs(exact);
  const unsigned rounding = (flags & RJ_ARITH_FP_ROUND_MASK) >> 24;
  const bool outward = rounding == (exact < 0 ? 2U : 1U);
  const bool overflow = rounding == 0 ? magnitude >= 0x1.ffffffp127
                        : outward     ? magnitude > 0x1.fffffep127
                                      : magnitude >= 0x1p128;
  if (overflow) {
    causes |= RJ_ARITH_EXCEPTION_OVERFLOW |
              ((modifiers & RJ_ARITH_ALU_OMOD_MASK) ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
  }
  return causes;
}

// Retain shared/alu_exceptions.h's native-nearest MUL classifier for RDNA3 and gfx1250.
// Neither has a capture behind the gfx1201 rule -- gfx1100 TRAPSTS does not latch with
// exceptions disabled, and gfx1250's inherited Compatibility policy reports range-only MUL
// causes -- so their guests keep the causes they saw before it, and a trap handler they
// reach today they still reach. The classifier observes raw operands, independently of the
// guest flush/rounding applied to the numeric result.
inline uint32_t compatibility_mul_causes(uint32_t a, uint32_t b, uint32_t omod, bool gfx12) {
  const uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff;
  const float x = as_float(a), y = as_float(b);
  const float scale = omod == 1 ? 2.0f : omod == 2 ? 4.0f : omod == 3 ? 0.5f : 1.0f;
  const double exact = double(x) * double(y) * scale;
  const float result = (x * y) * scale;
  const uint32_t magnitude = as_bits(result) & 0x7fffffff;
  uint32_t causes =
      ((ma && ma < 0x800000) || (mb && mb < 0x800000)) ? RJ_ARITH_EXCEPTION_INPUT_DENORM : 0;
  if (ma < 0x7f800000 && mb < 0x7f800000 && magnitude == 0x7f800000) {
    causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
  }
  if (!(gfx12 && omod)) {
    if (exact != 0 && magnitude < 0x800000) {
      causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
    if (__builtin_isfinite(exact) && double(result) != exact) {
      causes |= RJ_ARITH_EXCEPTION_INEXACT;
    }
  }
  return causes;
}

// Inspect participating source lanes before any possibly aliased destination writes.
// On RDNA4 MUL uses the hardware classifier shared with explicit reporting
// (fp/mul_exceptions.h): range causes follow participating EXEC lanes, while DPP row/bank
// controls that suppress a destination also suppress that lane's INVALID, as in the inherited
// model. RDNA3 and gfx1250 keep the inherited classifier until captures establish the flags.
template <Binary Op, bool Rdna3, bool Gfx1250 = false>
inline uint32_t compatibility_binary_causes(uint64_t flags, uint32_t exec_mask, uint32_t write_mask,
                                            uint32_t modifiers, const uint32_t *a,
                                            const uint32_t *b) {
  uint32_t causes = 0;
  for (int lane = 0; lane < 32; ++lane) {
    if (!(exec_mask & (1U << lane))) {
      continue;
    }
    if constexpr (Op == Binary::Mul && (Rdna3 || Gfx1250)) {
      const uint32_t x = alu_input_bits(a[lane], modifiers);
      const uint32_t y = alu_input_bits(b[lane], modifiers >> 1);
      causes |= compatibility_mul_causes(x, y, (modifiers >> 6) & 3, Gfx1250);
    } else if constexpr (Op == Binary::Mul) {
      uint32_t lane_causes = mul_exceptions_f32(a[lane], b[lane], flags, modifiers);
      if (!(write_mask & (1U << lane))) {
        lane_causes &= ~RJ_ARITH_EXCEPTION_INVALID;
      }
      causes |= lane_causes;
    } else if constexpr (Op == Binary::Add || Op == Binary::Sub || Op == Binary::Subrev) {
      if constexpr (!Rdna3 && !Gfx1250) {
        if (write_mask & (1U << lane)) {
          causes |= add_exceptions_f32<Op>(a[lane], b[lane], flags, modifiers);
        }
      }
    } else if constexpr (!Rdna3 && !Gfx1250) {
      if (write_mask & (1U << lane)) {
        causes |= gfx12_binary_causes<Op>(flags, modifiers, a[lane], b[lane]);
      }
    }
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
