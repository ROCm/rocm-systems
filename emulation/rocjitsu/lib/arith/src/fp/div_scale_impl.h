// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division operand rescaling and its per-lane scaling-condition output.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Pre-scaling decisions adapted from rocjitsu shared/division.h.

#pragma once

#include "common/bits.h"
#include "fp/float_output.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

#include "fp/div_scale.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

template <int Width> struct DivisionScaleResult {
  typename IeeeFormat<Width>::Bits value;
  bool post_scale;
};

// Pre-scale A (equal to B or C after NEG modifiers), with B the denominator
// and C the numerator. Returns the scaled value and post-scaling condition.
// Uses guest rounding with guest denormal controls and preserves host FP state.
template <int Width>
inline DivisionScaleResult<Width>
division_scale(typename IeeeFormat<Width>::Bits a, typename IeeeFormat<Width>::Bits b,
               typename IeeeFormat<Width>::Bits c, uint32_t mode, bool quiet_nan, uint64_t flags) {
  using F = IeeeFormat<Width>;
  using T = typename F::Bits;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  if (mode & RJ_ARITH_ALU_NEG_A) {
    a ^= F::sign;
  }
  if (mode & RJ_ARITH_ALU_NEG_B) {
    b ^= F::sign;
  }
  if (mode & RJ_ARITH_ALU_NEG_C) {
    c ^= F::sign;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if ((a & F::infinity) == 0) {
      a &= F::sign;
    }
    if ((b & F::infinity) == 0) {
      b &= F::sign;
    }
    if ((c & F::infinity) == 0) {
      c &= F::sign;
    }
  }
  T bm = b & (F::sign - 1), cm = c & (F::sign - 1);
  int de = int((b & F::infinity) >> F::fraction), ne = int((c & F::infinity) >> F::fraction),
      delta = ne - de;
  int adjustment = 0;
  bool post_scale = false, is_denominator = a == b;
  if (delta >= threshold) {
    adjustment = is_denominator ? scale : 0;
    post_scale = true;
  } else if (de == 0) {
    adjustment = scale;
  } else if (de >= 2 * F::bias - 1) {
    post_scale = delta <= -threshold;
    adjustment = post_scale && !is_denominator ? 0 : -scale;
  } else if (delta <= -threshold) {
    adjustment = is_denominator ? 0 : scale;
    post_scale = true;
  } else if (ne <= F::fraction + 1) {
    adjustment = scale;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  T value;
  if (!bm || !cm) {
    value = F::sign | F::infinity | F::quiet;
  } else if (!quiet_nan && (a & (F::sign - 1)) > F::infinity) {
    value = a;
  } else {
    value = division_scale_bits<Width>(a, adjustment, rounding);
  }
  if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (value & F::infinity) == 0) {
    value &= F::sign;
  }
  if ((flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (value & (F::sign - 1)) > F::infinity) {
    mode &= ~RJ_ARITH_ALU_CLAMP;
  }
  return {
      float_output<Width>(value, mode, division_overflow<Width>((value & F::sign) != 0, rounding)),
      post_scale};
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
