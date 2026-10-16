// SPDX-License-Identifier: MIT

// FP32 two/three-input selection with gfx11/gfx12 NaN and signed-zero rules.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/float_bits.h"
#include "common/internal.h"

#include <stdint.h>

namespace rocjitsu::arith {

// IEEE selection orders -0 below +0. Number variants ignore even signaling NaNs
// when the other operand is numeric; propagating variants prefer signaling NaNs.
template <bool Maximum, bool Propagate, typename Float> Float minmax(Float x, Float y) {
  using Bits = FloatBits<Float>;
  typename Bits::UInt a = Bits::bits(x), b = Bits::bits(y);
  bool an = (a & Bits::magnitude) > Bits::infinity, bn = (b & Bits::magnitude) > Bits::infinity;
  if constexpr (Propagate) {
    if (an && !(a & Bits::quiet)) {
      return Bits::value(a | Bits::quiet);
    }
    if (bn && !(b & Bits::quiet)) {
      return Bits::value(b | Bits::quiet);
    }
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
  bool an = (a & Bits::magnitude) > Bits::infinity;
  bool bn = (b & Bits::magnitude) > Bits::infinity;
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

// Select A/B first and then C; median follows the ISA's first-maximum removal
// rule by default. OrderedMedian orders -0 below +0 for FP16. Either median
// returns minimumNumber(A,B,C) if any input is NaN.
// Return the selected three-input minimum, maximum, median, or combined selection with GPU NaN
// rules.
template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool OrderedMedian = false>
float minmax3_value(float x, float y, float z) {
  float value;
  if constexpr (Median) {
    const bool has_nan = (rocjitsu::arith::as_bits(x) & 0x7fffffff) > 0x7f800000 ||
                         (rocjitsu::arith::as_bits(y) & 0x7fffffff) > 0x7f800000 ||
                         (rocjitsu::arith::as_bits(z) & 0x7fffffff) > 0x7f800000;
    if (has_nan) {
      value = rocjitsu::arith::minmax<false, false>(rocjitsu::arith::minmax<false, false>(x, y), z);
    } else if constexpr (OrderedMedian) {
      value = rocjitsu::arith::minmax<true, false>(
          rocjitsu::arith::minmax<false, false>(x, y),
          rocjitsu::arith::minmax<false, false>(rocjitsu::arith::minmax<true, false>(x, y), z));
    } else {
      float maximum =
          rocjitsu::arith::minmax<true, false>(rocjitsu::arith::minmax<true, false>(x, y), z);
      value = maximum == x   ? rocjitsu::arith::minmax<true, false>(y, z)
              : maximum == y ? rocjitsu::arith::minmax<true, false>(x, z)
                             : rocjitsu::arith::minmax<true, false>(x, y);
    }
  } else {
    float ab = rocjitsu::arith::minmax<FirstMaximum, Propagate>(x, y);
    value = rocjitsu::arith::minmax<SecondMaximum, Propagate>(ab, z);
  }
  return value;
}

// RDNA3 ternary selection composes the binary IEEE-mode rule. Median removes
// the first maximum, except that any NaN selects the chained minimum instead.
template <bool FirstMaximum, bool SecondMaximum, bool Median>
float minmax3_rdna3(float x, float y, float z, bool ieee) {
  if constexpr (Median) {
    if ((as_bits(x) & 0x7fffffffU) > 0x7f800000U || (as_bits(y) & 0x7fffffffU) > 0x7f800000U ||
        (as_bits(z) & 0x7fffffffU) > 0x7f800000U) {
      return minmax_rdna3<false>(minmax_rdna3<false>(x, y, ieee), z, ieee);
    }
    float maximum = minmax_rdna3<true>(minmax_rdna3<true>(x, y, ieee), z, ieee);
    return maximum == x   ? minmax_rdna3<true>(y, z, ieee)
           : maximum == y ? minmax_rdna3<true>(x, z, ieee)
                          : minmax_rdna3<true>(x, y, ieee);
  }
  return minmax_rdna3<SecondMaximum>(minmax_rdna3<FirstMaximum>(x, y, ieee), z, ieee);
}

enum class Minmax3 {
  Min3Legacy,
  Max3Legacy,
  MinmaxLegacy,
  MaxminLegacy,
  MedianLegacy,
  MedianNum,
  Min3Num,
  Max3Num,
  MinmaxNum,
  MaxminNum,
  Minimum3,
  Maximum3,
  MinimumMaximum,
  MaximumMinimum,
};

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Select FP32 minima, maxima, or medians. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void minmax3_x86_64_v3(Minmax3 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

} // namespace rocjitsu::arith
