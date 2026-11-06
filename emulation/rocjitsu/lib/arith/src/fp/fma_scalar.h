// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP32 FMA semantics, including special values and instruction modifiers.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/alu.h"
#include "common/bits.h"
#include "fp/fma_integer.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Round a finite round-to-odd FP64 intermediate to FP32 using guest MODE.
// Flushing tests tininess after significand rounding, before exponent packing.
inline uint32_t fma_narrow_f32(uint64_t bits, bool flush_output, unsigned rounding = 0) {
  uint32_t sign = uint32_t(bits >> 32) & 0x80000000;
  uint64_t magnitude = bits & 0x7fffffffffffffffULL;
  if (!magnitude) {
    return sign;
  }
  const bool away = rounding == (sign ? 2u : 1u);
  const uint32_t overflow = sign | ((rounding == 0 || away) ? 0x7f800000 : 0x7f7fffff);
  int exponent = int(magnitude >> 52) - 1023;
  if (exponent > 127) {
    return overflow;
  }
  if (flush_output && exponent < -127) {
    return sign;
  }
  if (exponent < -150) {
    return sign | unsigned(away);
  }
  int shift = 29 + (!flush_output && exponent < -126 ? -126 - exponent : 0);
  uint64_t significand = (magnitude & 0xfffffffffffffULL) | (1ULL << 52);
  uint32_t rounded = uint32_t(significand >> shift);
  uint64_t unit = 1ULL << shift, remainder = significand & (unit - 1);
  rounded += rounding == 0 ? remainder > unit / 2 || (remainder == unit / 2 && (rounded & 1))
                           : away && remainder != 0;
  if (!flush_output && exponent < -126) {
    return sign | rounded;
  }
  if (rounded == (1U << 24)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > 127) {
    return overflow;
  }
  // With output flushing, an unpromoted tiny significand becomes signed zero.
  if (exponent < -126) {
    return sign;
  }
  return sign | (uint32_t(exponent + 127) << 23) | (rounded & 0x7fffff);
}

// rocjitsu shared/fp_mode.h FMA NaN ordering and guest rounding,
// using the existing integer intermediate instead of host FP-control changes.
// Inputs have already been decoded, modified and flushed by their source type.
inline uint32_t fma_value_f32(uint32_t a, uint32_t b, uint32_t c, uint64_t flags, bool rdna3) {
  const bool quiet_nan = !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE);
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;

  uint32_t result;
  if ((ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000)) {
    result = 0xffc00000;
  } else if (ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000) {
    result = (ma > 0x7f800000 ? a : mb > 0x7f800000 ? b : c) | (quiet_nan ? 0x00400000 : 0);
  } else if (ma == 0x7f800000 || mb == 0x7f800000) {
    result = ((a ^ b) & 0x80000000) | 0x7f800000;
    if (mc == 0x7f800000 && ((result ^ c) & 0x80000000)) {
      result = 0xffc00000;
    }
  } else if (mc == 0x7f800000) {
    result = c;
  } else {
    uint64_t value = fma_finite_round_odd(a, b, c);
    const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);

    // Exact cancellation yields -0 only for downward rounding. Matching
    // signed-zero terms retain their sign, as in rocjitsu's directed FMA model.
    const bool matching_zeros = (ma == 0 || mb == 0) && mc == 0 && ((a ^ b ^ c) & 0x80000000) == 0;
    if (!(value & 0x7fffffffffffffffULL) && !matching_zeros) {
      value = uint64_t(rounding == 2) << 63;
    }
    result = fma_narrow_f32(value, flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, rounding);
  }
  return result;
}

// Clamp an FP32 result to [0,1], retaining NaNs only when DX10 clamping is disabled.
inline uint32_t fma_clamp_f32(uint32_t value, uint64_t flags) {
  return clamp_f32_bits(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
}

// Apply source modifiers, guest rounding, OMOD and CLAMP for ordinary FP32 FMA.
[[gnu::always_inline]] inline uint32_t fma_instruction_value(uint32_t a, uint32_t b, uint32_t c,
                                                             uint64_t flags,
                                                             uint32_t instruction_flags, bool rdna3,
                                                             bool dx9_zero) {
  unsigned omod = (instruction_flags >> 6) & 3;
  uint32_t input[] = {a, b, c};
  for (int operand = 0; operand < 3; ++operand) {
    input[operand] = alu_input_bits(input[operand], instruction_flags >> operand, flags);
  }
  if (dx9_zero) {
    if (!(input[0] & 0x7fffffff) || !(input[1] & 0x7fffffff)) {
      // DX9 supplies a positive zero product before adding C.
      input[0] = 0;
      input[1] = 0x3f800000;
    }
  }
  // OMOD flushes a tiny intermediate before scaling, independently of MODE.
  uint32_t value = rocjitsu::arith::fma_value_f32(
      input[0], input[1], input[2], flags | (omod ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0), rdna3);
  value = rocjitsu::arith::scale_omod_f32_bits(value, omod,
                                               unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));

  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    value = fma_clamp_f32(value, flags);
  }
  return value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
