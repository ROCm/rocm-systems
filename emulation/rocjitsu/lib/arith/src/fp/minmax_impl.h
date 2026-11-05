// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local FP16/FP32/FP64 selection implementing the inherited selection model.
// These NaN encoding and zero-sign choices exceed the loose API guarantees.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "common/internal.h"
#include "fp/minmax.h"
#include "fp/selection_bits.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// This selection model orders -0 below +0. Number variants ignore even signaling NaNs
// when the other operand is numeric; propagating variants quiet the first NaN in source order.
template <bool Maximum, bool PropagateNaN, typename Float> Float minmax(Float x, Float y) {
  using Bits = FloatBits<Float>;
  typename Bits::UInt a = Bits::bits(x), b = Bits::bits(y);
  bool an = Bits::is_nan(a), bn = Bits::is_nan(b);
  if constexpr (PropagateNaN) {
    if (an || bn) {
      return Bits::value((an ? a : b) | Bits::quiet);
    }
  } else {
    if (an) {
      return bn ? Bits::value(a | Bits::quiet) : y;
    }
    if (bn) {
      return x;
    }
  }
  if (x == y) {
    return Bits::value(Maximum ? (a & b) : (a | b));
  }
  return (Maximum ? x > y : x < y) ? x : y;
}

// RDNA3 MIN/MAX prefer a signaling NaN only in IEEE mode. Otherwise a lone
// NaN loses to a number, while two NaNs return A without quieting it.
template <bool Maximum, typename Float> Float minmax_rdna3(Float x, Float y, bool ieee) {
  using Bits = FloatBits<Float>;
  auto a = Bits::bits(x), b = Bits::bits(y);
  bool an = Bits::is_nan(a);
  bool bn = Bits::is_nan(b);
  if (ieee) {
    if (an && !(a & Bits::quiet)) {
      return Bits::value(a | Bits::quiet);
    }
    if (bn && !(b & Bits::quiet)) {
      return Bits::value(b | Bits::quiet);
    }
  } else if (an && bn) {
    return x;
  }
  return minmax<Maximum, false>(x, y);
}

// Select A/B then C. Median orders -0 below +0 and returns the chained
// minimumNumber if any input is NaN.
template <bool FirstMaximum, bool SecondMaximum, bool PropagateNaN, bool Median = false>
float minmax3_value(float x, float y, float z) {
  float value;
  if constexpr (Median) {
    const bool has_nan = FloatBits<float>::is_nan(FloatBits<float>::bits(x)) ||
                         FloatBits<float>::is_nan(FloatBits<float>::bits(y)) ||
                         FloatBits<float>::is_nan(FloatBits<float>::bits(z));
    if (has_nan) {
      value = rocjitsu::arith::minmax<false, false>(rocjitsu::arith::minmax<false, false>(x, y), z);
    } else {
      value = rocjitsu::arith::minmax<true, false>(
          rocjitsu::arith::minmax<false, false>(x, y),
          rocjitsu::arith::minmax<false, false>(rocjitsu::arith::minmax<true, false>(x, y), z));
    }
  } else {
    float ab = rocjitsu::arith::minmax<FirstMaximum, PropagateNaN>(x, y);
    value = rocjitsu::arith::minmax<SecondMaximum, PropagateNaN>(ab, z);
  }
  return value;
}

// RDNA3 median orders -0 below +0, as in rocjitsu shared/minmax.h.
// Any NaN selects the chained minimum using the binary IEEE-mode rule.
template <bool FirstMaximum, bool SecondMaximum, bool Median>
float minmax3_rdna3(float x, float y, float z, bool ieee) {
  if constexpr (Median) {
    if (FloatBits<float>::is_nan(FloatBits<float>::bits(x)) ||
        FloatBits<float>::is_nan(FloatBits<float>::bits(y)) ||
        FloatBits<float>::is_nan(FloatBits<float>::bits(z))) {
      return minmax_rdna3<false>(minmax_rdna3<false>(x, y, ieee), z, ieee);
    }
    return minmax_rdna3<true>(minmax_rdna3<false>(x, y, ieee),
                              minmax_rdna3<false>(minmax_rdna3<true>(x, y, ieee), z, ieee), ieee);
  }
  return minmax_rdna3<SecondMaximum>(minmax_rdna3<FirstMaximum>(x, y, ieee), z, ieee);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
