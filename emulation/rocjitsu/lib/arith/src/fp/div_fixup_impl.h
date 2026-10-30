// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division-result fixup, special values, and optional guest exception causes.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Adapted from rocjitsu shared/division.h; FP16 saturation and OMOD behavior
// follow RX 9070 captures rather than the promoted-FP32 rocjitsu implementation.

#pragma once

#include "common/bits.h"
#include "fp/float_output.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

#include "fp/div_fixup.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Repair provisional quotient P using denominator B and numerator C, preserving
// denormals and applying guest-directed overflow recovery. FP16 output narrowing
// remains nearest-even. Applies source/output modifiers and FP16 saturation
// without reading or changing host FP state.
template <int Width, bool Rdna3 = false>
inline typename IeeeFormat<Width>::Bits
fixup_value(typename IeeeFormat<Width>::Bits p, typename IeeeFormat<Width>::Bits b,
            typename IeeeFormat<Width>::Bits c, uint32_t mode, bool saturate, unsigned rounding = 0,
            bool flush_output = false, bool dx10_clamp_off = false) {
  using F = IeeeFormat<Width>;
  using T = typename F::Bits;
  T inputs[] = {p, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
      inputs[i] &= F::sign - 1;
    }
    if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
      inputs[i] ^= F::sign;
    }
  }
  p = inputs[0];
  b = inputs[1];
  c = inputs[2];
  T bm = b & (F::sign - 1), cm = c & (F::sign - 1), pm = p & (F::sign - 1),
    sign = (b ^ c) & F::sign, result;
  T overflow = Width == 16 ? F::infinity - saturate : division_overflow<Width>(sign != 0, rounding);
  if (cm > F::infinity) {
    result = c | F::quiet;
  } else if (bm > F::infinity) {
    result = b | F::quiet;
  } else if ((bm == 0 && cm == 0) || (bm == F::infinity && cm == F::infinity)) {
    result = F::sign | F::infinity | F::quiet;
  } else if (bm == 0 || cm == F::infinity) {
    result = sign | F::infinity;
  } else if (cm == 0 || bm == F::infinity) {
    result = sign;
  } else if (Width != 16 &&
             int(cm >> F::fraction) - int(bm >> F::fraction) < -(F::bias + F::fraction)) {
    // rocjitsu div_fixup rounds its extreme-underflow recovery with div_round.
    // It is below half the smallest subnormal: only rounding away from zero
    // can retain one, and guest output flushing then removes it.
    result = sign | T(!flush_output && rounding == (sign ? 2U : 1U));
  } else {
    // rocjitsu div_fixup_f16 selects infinity or max-half before promoted
    // scaling and nearest-even narrowing. RDNA4 additionally saturates this
    // provisional overflow, as established by the existing FP16_OVFL captures.
    T recovery = division_overflow<Width>(sign != 0, rounding);
    if (Width == 16 && !Rdna3 && saturate) {
      recovery = F::infinity - 1;
    }
    result = sign | (pm >= F::infinity ? recovery : pm);
  }
  if constexpr (Rdna3 && Width == 16) {
    static_assert(Width == 16);

    // Port the generated RDNA3 handler's promoted-F32 scale, half narrowing
    // and finalize_omod_f16 stages using integer IEEE bits.
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      bool finite = (result & (F::sign - 1)) < F::infinity;
      result = division_scale_bits<16>(result, omod == 3 ? -1 : int(omod));
      if (saturate && finite && (result & (F::sign - 1)) == F::infinity) {
        result = (result & F::sign) | (F::infinity - 1);
      }
      if (!(result & F::infinity)) {
        result = 0;
      }
    }
    if (dx10_clamp_off && (result & (F::sign - 1)) > F::infinity) {
      return result;
    }
    return float_output<16>(result, mode & RJ_ARITH_ALU_CLAMP);
  }
  if (dx10_clamp_off && (result & (F::sign - 1)) > F::infinity) {
    mode &= ~RJ_ARITH_ALU_CLAMP;
  }
  return float_output<Width>(result, mode, overflow);
}

// Preserve the pre-migration RDNA3 classifier (shared/alu_exceptions.h).
// DX10_CLAMP disables this classifier. Operands here have input flushing applied.
template <int Width>
inline uint32_t
fixup_rdna3_exceptions(typename IeeeFormat<Width>::Bits p, typename IeeeFormat<Width>::Bits b,
                       typename IeeeFormat<Width>::Bits c, uint32_t mode, uint64_t flags) {
  using F = IeeeFormat<Width>;
  if (!(flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return 0;
  }
  auto d = b & (F::sign - 1), n = c & (F::sign - 1);
  uint32_t causes = ((d && d < (typename F::Bits(1) << F::fraction)) ||
                     (n && n < (typename F::Bits(1) << F::fraction)))
                        ? RJ_ARITH_EXCEPTION_INPUT_DENORM
                        : 0;
  if ((d > F::infinity && !(d & F::quiet)) || (n > F::infinity && !(n & F::quiet)) || (!d && !n) ||
      (d == F::infinity && n == F::infinity)) {
    causes |= RJ_ARITH_EXCEPTION_INVALID;
  } else if (!d && n && n < F::infinity) {
    causes |= RJ_ARITH_EXCEPTION_FLOAT_DIV0;
  }
  if (d && d < F::infinity && n && n < F::infinity) {
    if (int(n >> F::fraction) - int(d >> F::fraction) < -(F::bias + F::fraction)) {
      causes |= RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    } else if ((p & (F::sign - 1)) >= F::infinity) {
      causes |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    }
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    causes &= ~(RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT);
  }
  if (omod == 1 || omod == 2) {
    const auto value =
        fixup_value<Width, true>(p, b, c, mode & ~(RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP),
                                 false, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24),
                                 flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, true) &
        (F::sign - 1);
    if (value < F::infinity && int(value >> F::fraction) + int(omod) >= 2 * F::bias + 1) {
      causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
    }
  }
  return causes;
}

// Classify DIV_FIXUP's wave exception contribution from unmodified source bits.
// Adapted from rocjitsu shared/alu_exceptions.h, with RX 9070 corrections:
// CLAMP suppresses all flags; NaN/invalid cases suppress input-denormal flags;
// FP16 has no exponent-gap underflow recovery. Signs do not affect these flags.
template <int Width>
inline uint32_t fixup_exceptions(typename IeeeFormat<Width>::Bits p,
                                 typename IeeeFormat<Width>::Bits b,
                                 typename IeeeFormat<Width>::Bits c, uint32_t mode) {
  using F = IeeeFormat<Width>;
  p &= F::sign - 1;
  b &= F::sign - 1;
  c &= F::sign - 1;
  if (mode & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }
  if ((b > F::infinity && !(b & F::quiet)) || (c > F::infinity && !(c & F::quiet))) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  if (b > F::infinity || c > F::infinity) {
    return 0;
  }
  if ((!b && !c) || (b == F::infinity && c == F::infinity)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  if (!b && c && c < F::infinity) {
    return RJ_ARITH_EXCEPTION_FLOAT_DIV0;
  }
  uint32_t exceptions = ((b && b < (1ULL << F::fraction)) || (c && c < (1ULL << F::fraction)))
                            ? RJ_ARITH_EXCEPTION_INPUT_DENORM
                            : 0;
  if (!b || !c || b >= F::infinity || c >= F::infinity) {
    return exceptions;
  }
  unsigned omod = (mode >> 6) & 3;
  if (Width != 16 && int(c >> F::fraction) - int(b >> F::fraction) < -(F::bias + F::fraction)) {
    return exceptions | (omod ? 0 : RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT);
  }
  if (p >= F::infinity) {
    return exceptions | RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
  }
  if ((omod == 1 || omod == 2) && (p >> F::fraction) + omod >= unsigned(2 * F::bias + 1)) {
    exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW;
  }
  return exceptions;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
