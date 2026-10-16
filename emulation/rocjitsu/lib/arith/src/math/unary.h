// SPDX-License-Identifier: MIT

// FP32 unary math, integral rounding, and mantissa decomposition.
// Internal helpers shared by instruction implementations; not part of the public C API.
// rndne is adapted from rocjitsu util::rndne_scalar (MIT).
// frexp_mant_f32_bits uses amdgpu::frexp_f32 (ffc144c564c), with raw bits.

#pragma once

#include "common/bits.h"
#include "common/internal.h"
#include "common/round_integral.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

enum class Unary { Trunc, Ceil, Rndne, Floor, Sqrt, Rcp, Rsq, Exp, Log, Fract, FrexpMant };

template <Unary Op>
inline constexpr bool unary_flushes_f32 = Op == Unary::Sqrt || Op == Unary::Rcp ||
                                          Op == Unary::Rsq || Op == Unary::Exp || Op == Unary::Log;

template <Unary Op>
inline constexpr bool unary_rounds_f32 =
    Op == Unary::Trunc || Op == Unary::Ceil || Op == Unary::Rndne || Op == Unary::Floor;

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
  unsigned shift = exponent ? 0 : bit_count_zero<false>(magnitude) - 8;
  return (bits & 0x80000000) | 0x3f000000 | ((magnitude << shift) & 0x7fffff);
}

// Return the selected loose unary math result for one value or vector of values.
template <Unary Op> inline float unary_value(float value) {
  if constexpr (Op == Unary::FrexpMant) {
    return as_float(frexp_mant_f32_bits(as_bits(value), false));
  }
  if constexpr (Op == Unary::Trunc) {
    return std::trunc(value);
  }
  if constexpr (Op == Unary::Ceil) {
    return std::ceil(value);
  }
  if constexpr (Op == Unary::Rndne) {
    return rndne(value);
  }
  if constexpr (Op == Unary::Floor) {
    return std::floor(value);
  }
  if constexpr (Op == Unary::Sqrt) {
    return std::sqrt(value);
  }
  if constexpr (Op == Unary::Rcp) {
    return 1.0f / value;
  }
  if constexpr (Op == Unary::Rsq) {
    return 1.0f / std::sqrt(value);
  }
  if constexpr (Op == Unary::Exp) {
    return std::exp2(value);
  }
  if constexpr (Op == Unary::Fract) {
    float result = value - std::floor(value);
    return result > as_float(0x3f7fffff) ? as_float(0x3f7fffff) : result;
  }
  if constexpr (Op == Unary::Log) {
    return std::log2(value);
  }
}

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate FP32 unary math. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void unary_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                     const uint32_t *a, uint64_t flags);

// Integral rounding for all FP32 values, including NaN payload preservation.
void round_exact_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate FP32 unary math. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void round_exact_x86_64_v4(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags);
#endif

} // namespace rocjitsu::arith
