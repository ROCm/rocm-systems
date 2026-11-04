// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IEEE format traits and bitwise OMOD/CLAMP shared by arithmetic and selection.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Format and rounding helpers adapted from rocjitsu shared/division.h.

#pragma once

#include "common/bits.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

template <int Width> struct IeeeFormat {
  using Bits = std::conditional_t<Width == 64, uint64_t, uint32_t>;
  static constexpr int fraction = Width == 16 ? 10 : Width == 32 ? 23 : 52;
  static constexpr int bias = Width == 16 ? 15 : Width == 32 ? 127 : 1023;
  static constexpr Bits sign = Bits(1) << (Width - 1);
  static constexpr Bits infinity = Bits(2 * bias + 1) << fraction;
  static constexpr Bits quiet = Bits(1) << (fraction - 1);
};

// Apply OMOD/CLAMP to IEEE bits. Active OMOD flushes input subnormals to +0
// and output subnormals to signed zero. Finite overflow uses the supplied code.
template <int Width>
inline typename IeeeFormat<Width>::Bits
float_output(typename IeeeFormat<Width>::Bits result, uint32_t instruction_flags,
             typename IeeeFormat<Width>::Bits overflow = IeeeFormat<Width>::infinity) {
  using F = IeeeFormat<Width>;
  using T = typename F::Bits;
  unsigned omod = (instruction_flags >> 6) & 3;
  if (omod) {
    T magnitude = result & (F::sign - 1), out_sign = result & F::sign;
    int exponent = int((result & F::infinity) >> F::fraction);
    if (!exponent) {
      result = 0;
    } else if (magnitude < F::infinity) {
      exponent += omod == 3 ? -1 : int(omod);
      if (exponent <= 0) {
        result = out_sign;
      } else if (exponent >= 2 * F::bias + 1) {
        result = out_sign | overflow;
      } else {
        result = (result & ~F::infinity) | (T(exponent) << F::fraction);
      }
    }
  }
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    if ((result & F::sign) || (result > F::infinity)) {
      result = 0;
    } else if (result > (T(F::bias) << F::fraction)) {
      result = T(F::bias) << F::fraction;
    }
  }
  return result;
}

// Apply selection output modifiers with guest overflow rounding and FP16 saturation.
// Preserve selected NaN payloads; DX10_CLAMP_OFF disables clamping a NaN.
template <int Width>
inline typename IeeeFormat<Width>::Bits selection_output(typename IeeeFormat<Width>::Bits bits,
                                                         uint32_t instruction_flags,
                                                         uint64_t flags) {
  using F = IeeeFormat<Width>;
  unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  bool infinity = rounding == 0 || rounding == (bits & F::sign ? 2U : 1U);
  if constexpr (Width == 16) {
    infinity &= !(flags & RJ_ARITH_FP16_OVFL);
  }
  if ((flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (bits & (F::sign - 1)) > F::infinity) {
    instruction_flags &= ~RJ_ARITH_ALU_CLAMP;
  }
  return float_output<Width>(bits, instruction_flags, F::infinity - !infinity);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
