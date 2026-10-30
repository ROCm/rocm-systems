// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 operand modifiers, output scaling/clamping, and guest denormal rules.
// Also owns the precision-independent gfx11 OMOD gate.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// gfx11 disables OMOD under IEEE mode or when output denormals are preserved.
inline uint64_t rdna3_output_modifiers(uint64_t flags, uint64_t instruction_flags) {
  return !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)
             ? instruction_flags & ~uint64_t(RJ_ARITH_ALU_OMOD_MASK)
             : instruction_flags;
}

// Apply ABS then NEG and guest input flushing to FP32 bits. Modifiers are
// shifted to operand A's bit positions; flushing preserves the resulting sign.
inline uint32_t alu_input_bits(uint32_t bits, uint32_t modifiers, uint64_t flags = 0) {
  if (modifiers & RJ_ARITH_ALU_ABS_A) {
    bits &= 0x7fffffff;
  }
  if (modifiers & RJ_ARITH_ALU_NEG_A) {
    bits ^= 0x80000000;
  }
  if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && !(bits & 0x7f800000)) {
    bits &= 0x80000000;
  }
  return bits;
}

// Replace FP32 subnormals with zero of the same sign.
inline float flush_denorm_f32(float value) {
  return as_float(alu_input_bits(as_bits(value), 0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS));
}

// Apply ABS then NEG to raw FP32 bits and return the operand value.
inline float alu_input(uint32_t bits, uint32_t modifiers) {
  return as_float(alu_input_bits(bits, modifiers));
}

// Clamp raw FP32 to [0, 1], optionally preserving NaNs under DX10_CLAMP_OFF.
inline uint32_t clamp_f32_bits(uint32_t bits, bool preserve_nan) {
  const bool nan = (bits & 0x7fffffff) > 0x7f800000;
  if (nan && preserve_nan) {
    return bits;
  }
  return (bits & 0x80000000) || nan ? 0 : bits > 0x3f800000 ? 0x3f800000 : bits;
}

// Apply OMOD scaling only; the caller handles CLAMP separately.
inline float scale_omod(float value, uint32_t modifiers) {
  switch ((modifiers >> 6) & 3) {
  case 1:
    value *= 2;
    break;
  case 2:
    value *= 4;
    break;
  case 3:
    value *= 0.5f;
    break;
  }
  return value;
}

// Scale a rounded FP32 result with architectural tiny-result flushing and
// directed overflow. Integer exponent adjustment is independent of host controls.
inline uint32_t scale_omod_f32_bits(uint32_t bits, unsigned omod, unsigned rounding) {
  const uint32_t magnitude = bits & 0x7fffffff;
  if (!omod || magnitude >= 0x7f800000) {
    return bits;
  }
  const unsigned exponent = magnitude >> 23;
  if (!exponent) {
    return 0;
  }
  if (omod == 3) {
    return exponent == 1 ? bits & 0x80000000 : bits - 0x800000;
  }
  if (exponent + omod >= 255) {
    const bool infinity = rounding == 0 || rounding == (bits >> 31 ? 2u : 1u);
    return (bits & 0x80000000) | (infinity ? 0x7f800000 : 0x7f7fffff);
  }
  return bits + (omod << 23);
}

// FP32 scaling follows rocjitsu's fp_mode::apply_omod_f32. Tiny unscaled
// results become +0; halving a normal result below twice minimum normal gives
// signed zero, independently of guest denormal mode.
inline float scale_omod_f32(float value, uint32_t modifiers) {
  unsigned omod = (modifiers >> 6) & 3;
  if (omod) {
    uint32_t raw = as_bits(value), magnitude = raw & 0x7fffffff;
    if (magnitude < 0x00800000) {
      value = 0.0f;
    } else if (omod == 3 && magnitude < 0x01000000) {
      value = as_float(raw & 0x80000000);
    }
  }
  return scale_omod(value, modifiers);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
