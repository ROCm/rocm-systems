// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division-result fixup, special values, and optional guest exception causes.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Adapted from rocjitsu shared/division.h; FP16 saturation and OMOD behavior
// follow RX 9070 captures rather than the promoted-FP32 rocjitsu implementation.

#pragma once

#include "fp/division.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Repair provisional quotient P using denominator B and numerator C, preserving
// denormals and applying guest-directed overflow recovery. FP16 output narrowing
// remains nearest-even. Applies source/output modifiers and FP16 saturation
// without reading or changing host FP state.
template <int Width, bool Rdna3Half = false>
inline typename DivisionFormat<Width>::Bits
fixup_value(typename DivisionFormat<Width>::Bits p, typename DivisionFormat<Width>::Bits b,
            typename DivisionFormat<Width>::Bits c, uint32_t mode, bool saturate,
            unsigned rounding = 0, bool flush_output = false, bool dx10_clamp_off = false) {
  using F = DivisionFormat<Width>;
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
    if (Width == 16 && !Rdna3Half && saturate) {
      recovery = F::infinity - 1;
    }
    result = sign | (pm >= F::infinity ? recovery : pm);
  }
  if constexpr (Rdna3Half) {
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
    return division_output<16>(result, mode & RJ_ARITH_ALU_CLAMP);
  }
  return division_output<Width>(result, mode, overflow);
}

// Classify DIV_FIXUP's wave exception contribution from unmodified source bits.
// Adapted from rocjitsu shared/alu_exceptions.h, with RX 9070 corrections:
// CLAMP suppresses all flags; NaN/invalid cases suppress input-denormal flags;
// FP16 has no exponent-gap underflow recovery. Signs do not affect these flags.
template <int Width>
inline uint32_t fixup_exceptions(typename DivisionFormat<Width>::Bits p,
                                 typename DivisionFormat<Width>::Bits b,
                                 typename DivisionFormat<Width>::Bits c, uint32_t mode) {
  using F = DivisionFormat<Width>;
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

// Fix up division results and special values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3Half = false>
void fixup_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

// Fix up division results and special values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3Half = false>
void fixup_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

// Wave contribution from the same operand layout as numeric FIXUP. CLAMP
// suppresses causes; otherwise only EXEC-selected lanes contribute.
template <int Width>
uint32_t fixup_exceptions_x86_64_v3(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Fix up division results and special values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
uint32_t fixup_exceptions_x86_64_v4(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

} // namespace rocjitsu::arith
