// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar exact unary evaluation, output modifiers, and optional FP status.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "common/round_integral.h"
#include "math/exp_model.h"
#include "math/log_model.h"
#include "math/rcp_model.h"
#include "math/rsq_model.h"
#include "math/sqrt_model.h"
#include "math/unary_impl.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Apply output flushing, OMOD and CLAMP to an inherited unary result.
inline uint32_t unary_exact_output(uint64_t flags, uint32_t mode, uint32_t value) {
  unsigned omod = (mode >> 6) & 3;
  if (!(value & 0x7f800000)) {
    value &= 0x80000000;
  }
  uint32_t magnitude = value & 0x7fffffff;
  if (omod && magnitude < 0x7f800000) {
    unsigned exponent = magnitude >> 23;
    if (!exponent) {
      value = 0;
    } else if (omod == 3 && exponent == 1) {
      value &= 0x80000000;
    } else if (omod != 3 && exponent + omod >= 255) {
      value = (value & 0x80000000) | 0x7f800000;
    } else {
      value = omod == 3 ? value - 0x800000 : value + (omod << 23);
    }
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) &&
      (!(flags & RJ_ARITH_FP_DX10_CLAMP_OFF) || (value & 0x7fffffff) <= 0x7f800000)) {
    value = (value & 0x80000000) || (value & 0x7fffffff) > 0x7f800000 ? 0
            : value > 0x3f800000                                      ? 0x3f800000
                                                                      : value;
  }
  return value;
}

// Evaluate one raw FP32 operand using the borrowed rocjitsu mappings and apply
// source/output modifiers. The caller supplies the effective architecture OMOD.
// All calculation uses integer bits, independently of host FP controls.
template <Unary Op>
[[gnu::always_inline]] inline uint32_t unary_exact_value(uint64_t flags, uint32_t mode,
                                                         uint32_t input, bool rdna3) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    input &= 0x7fffffff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    input ^= 0x80000000;
  }
  bool quiet_snan = !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE);
  uint32_t value;
  if constexpr (rocjitsu::arith::unary_rounds_f32<Op>) {
    constexpr auto rounding =
        Op == rocjitsu::arith::Unary::Trunc   ? rocjitsu::arith::IntegralRound::Trunc
        : Op == rocjitsu::arith::Unary::Ceil  ? rocjitsu::arith::IntegralRound::Ceil
        : Op == rocjitsu::arith::Unary::Floor ? rocjitsu::arith::IntegralRound::Floor
                                              : rocjitsu::arith::IntegralRound::Nearest;
    value = rocjitsu::arith::round_integral_bits<rounding, false>(
        input, flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  } else if constexpr (Op == rocjitsu::arith::Unary::FrexpMant) {
    value = rocjitsu::arith::frexp_mant_f32_bits(input, flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  } else if constexpr (Op == rocjitsu::arith::Unary::Rcp) {
    value = rocjitsu::arith::rcp_f32_bits(input, quiet_snan);
  } else if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
    value = rocjitsu::arith::sqrt_f32_bits(input, quiet_snan);
  } else if constexpr (Op == rocjitsu::arith::Unary::Rsq) {
    value = rocjitsu::arith::rsq_f32_bits(input, quiet_snan);
  } else if constexpr (Op == rocjitsu::arith::Unary::Exp) {
    value = rocjitsu::arith::exp_f32_bits(input, quiet_snan);
  } else {
    value = rocjitsu::arith::log_f32_bits(input, quiet_snan);
  }
  return unary_exact_output(flags, mode, value);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
