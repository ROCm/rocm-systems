// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 unary math, integral rounding, and mantissa decomposition.
// Internal helpers shared by instruction implementations; not part of the public C API.
// rndne is adapted from rocjitsu util::rndne_scalar (MIT).
// frexp_mant_f32_bits uses amdgpu::frexp_f32 (ffc144c564c), with raw bits.

#pragma once

#include "common/bits.h"
#include "common/internal.h"
#include "common/round_integral.h"
#include "math/unary.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Round to an integral FP32 value, ties to even, preserving signed zero and
// quieting NaNs without depending on the host rounding mode.
inline float rndne(float value) {
  return as_float(round_integral_bits<IntegralRound::Nearest, false>(as_bits(value), false));
}

// Return the signed mantissa with magnitude in [0.5, 1) for finite nonzero inputs. Preserve
// zeros/infinities, quiet signaling NaNs, and honor guest input flushing.
inline uint32_t frexp_mant_f32_bits(uint32_t bits, bool flush_input) {
  uint32_t magnitude = bits & 0x7fffffff;
  unsigned exponent = magnitude >> 23;
  if (exponent == 255) {
    return bits | (magnitude > 0x7f800000 ? 0x400000 : 0);
  }
  if (!magnitude || (!exponent && flush_input)) {
    return bits & 0x80000000;
  }
  unsigned shift = exponent ? 0 : rocjitsu::arith::countl_zero(magnitude) - 8;
  return (bits & 0x80000000) | 0x3f000000 | ((magnitude << shift) & 0x7fffff);
}

// Return the selected loose unary math result for one value or vector of values.
template <Unary Op> inline float unary_value(float value) {
  if constexpr (Op == Unary::FrexpMant) {
    return as_float(frexp_mant_f32_bits(as_bits(value), false));
  }
  if constexpr (Op == Unary::Trunc) {
    return __builtin_truncf(value);
  }
  if constexpr (Op == Unary::Ceil) {
    return __builtin_ceilf(value);
  }
  if constexpr (Op == Unary::Rndne) {
    return rndne(value);
  }
  if constexpr (Op == Unary::Floor) {
    return __builtin_floorf(value);
  }
  if constexpr (Op == Unary::Sqrt) {
    return __builtin_sqrtf(value);
  }
  if constexpr (Op == Unary::Rcp) {
    return 1.0f / value;
  }
  if constexpr (Op == Unary::Rsq) {
    return 1.0f / __builtin_sqrtf(value);
  }
  if constexpr (Op == Unary::Exp) {
    return __builtin_exp2f(value);
  }
  if constexpr (Op == Unary::Fract) {
    float result = value - __builtin_floorf(value);
    return result > as_float(0x3f7fffff) ? as_float(0x3f7fffff) : result;
  }
  if constexpr (Op == Unary::Log) {
    return __builtin_log2f(value);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
