// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared FP status classification of a rounded FP16/FP32 result against its exact value.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Hardware rule (gfx1201 EXCP_FLAG_USER captures of every SALU ADD/SUB/MUL/FMA form,
// S_CVT_F32_I32/U32 and S_CVT_F16_F32 across all 84 MODE rounding, denormal and FP16_OVFL
// settings, Nov 11 2026, 27.3M results; and V_MUL_F32 across the same MODEs, Nov 18 2026):
// overflow and tininess are judged after rounding the significand with the guest rounding
// mode and an unbounded exponent, so a result stored as the minimum normal still underflows
// when the unbounded rounding stays below it, and a saturated or directed finite result still
// overflows. Inexact compares the delivered result, including a flushed output, with the exact
// value. Classification uses integer intermediates, never host exception flags.

#pragma once

#include "common/bits.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen a finite FP16 (Half) or FP32 encoding to the FP64 encoding of the same value.
template <bool Half> inline uint64_t widen_bits_f64(uint32_t bits) {
  constexpr int fraction = Half ? 10 : 23, bias = Half ? 15 : 127;
  const uint64_t sign = uint64_t((bits >> (Half ? 15 : 31)) & 1) << 63;
  int exponent = int(bits >> fraction) & (Half ? 31 : 255);
  uint64_t significand = bits & ((1U << fraction) - 1);
  if (!exponent) {
    if (!significand) {
      return sign;
    }
    const int shift = rocjitsu::arith::countl_zero(significand) - (63 - fraction);
    significand = (significand << shift) & ((1U << fraction) - 1);
    exponent = 1 - shift;
  }
  return sign | (uint64_t(exponent - bias + 1023) << 52) | (significand << (52 - fraction));
}

// Classify one rounding of a nonzero exact value, given as FP64 bits, into FP16 (Half) or
// FP32. The FP64 value may be a round-to-odd intermediate with at least two extra bits.
// result is the delivered destination encoding before any output modifier.
template <bool Half>
inline uint32_t rounding_causes(uint64_t exact, uint32_t result, unsigned rounding) {
  constexpr int precision = Half ? 11 : 24;
  constexpr uint64_t largest = Half ? 0x40effc0000000000ULL : 0x47efffffe0000000ULL;
  constexpr uint64_t smallest_normal = Half ? 0x3f10000000000000ULL : 0x3810000000000000ULL;
  const uint64_t sign = exact & (1ULL << 63), unit = 1ULL << (53 - precision);
  uint64_t magnitude = exact & ~sign, remainder = magnitude & (unit - 1);
  magnitude -= remainder;
  const bool away = rounding == (sign ? 2U : 1U);
  if (rounding == 0 ? remainder > unit / 2 || (remainder == unit / 2 && (magnitude & unit))
                    : away && remainder) {
    magnitude += unit;
  }
  if (magnitude > largest) {
    return RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
  }
  if (widen_bits_f64<Half>(result) == exact) {
    return 0;
  }
  return RJ_ARITH_EXCEPTION_INEXACT |
         (magnitude < smallest_normal ? RJ_ARITH_EXCEPTION_UNDERFLOW : 0);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
