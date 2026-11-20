// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar exact FP64 reciprocal, reciprocal-square-root and square-root evaluation and output
// modifiers.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "fp/fp64.h"
#include "math/trans_f64_model.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Apply output flushing, OMOD and CLAMP to a TRANS-unit FP64 result. OMOD
// overflow gives infinity: these results never take guest rounding.
inline uint64_t trans_f64_exact_output(uint64_t flags, uint32_t mode, uint64_t value) {
  constexpr uint64_t sign = trans_f64::kSign, infinity = trans_f64::kInfinity;
  if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (value & ~sign) < trans_f64::kMinNormal) {
    value &= sign;
  }
  unsigned omod = (mode >> 6) & 3;
  uint64_t magnitude = value & ~sign;
  if (omod && magnitude < infinity) {
    unsigned exponent = unsigned(magnitude >> 52);
    if (!exponent) {
      value = 0;
    } else if (omod == 3 && exponent == 1) {
      value &= sign;
    } else if (omod != 3 && exponent + omod >= 2047) {
      value = (value & sign) | infinity;
    } else {
      value = omod == 3 ? value - trans_f64::kMinNormal : value + (uint64_t(omod) << 52);
    }
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) &&
      (!(flags & RJ_ARITH_FP_DX10_CLAMP_OFF) || (value & ~sign) <= infinity)) {
    value = (value & sign) || (value & ~sign) > infinity ? 0
            : value > 0x3ff0000000000000ULL              ? 0x3ff0000000000000ULL
                                                         : value;
  }
  return value;
}

// Evaluate one raw FP64 operand with the TRANS-unit model and apply source and
// output modifiers. The caller supplies the effective architecture OMOD.
// All calculation uses integer bits, independently of host FP controls.
template <Fp64 Op>
[[gnu::always_inline]] inline uint64_t trans_f64_exact_value(uint64_t flags, uint32_t mode,
                                                             uint64_t input, bool quiet_snan) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    input &= ~trans_f64::kSign;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    input ^= trans_f64::kSign;
  }
  if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
      (input & ~trans_f64::kSign) < trans_f64::kMinNormal) {
    input &= trans_f64::kSign;
  }
  return trans_f64_exact_output(flags, mode, trans_f64::trans_f64_bits<Op>(input, quiet_snan));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
