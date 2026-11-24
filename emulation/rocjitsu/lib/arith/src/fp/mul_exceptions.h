// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest FP32 multiplication exception classification from operands and controls.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/alu.h"
#include "common/bits.h"
#include "common/validation.h"
#include "fp/fma_integer.h"
#include "fp/fma_scalar.h"
#include "fp/rounding_status.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// The one V_MUL_F32 status classifier, shared by Compatibility and explicit reporting.
// Fitted to gfx1201 EXCP_FLAG_USER captures (R9700, Nov 18 2026) of V_MUL_F32 with and
// without ABS/NEG, OMOD and CLAMP across all 84 MODE settings (2.9M results, 0 mismatches):
// CLAMP suppresses every cause. NaN operands suppress input-denormal reporting; signaling
// NaNs and infinity times zero raise INVALID. Input denormals report only when preserved.
// Overflow and tininess follow the unbounded-exponent rule of rounding_causes. An output
// modifier suppresses only UNDERFLOW and INEXACT, and OVERFLOW is then raised when the
// product overflows before or after scaling; INVALID and INPUT_DENORM describe the operands
// and survive it. Finite FP32 products fit exactly in the FP64 intermediate.
inline uint32_t mul_exceptions_f32(uint32_t a, uint32_t b, uint64_t flags, uint32_t modifiers) {
  if (modifiers & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }
  a = alu_input_bits(a, modifiers);
  b = alu_input_bits(b, modifiers >> 1);
  uint32_t ma = a & 0x7fffffffU, mb = b & 0x7fffffffU;
  // Normal operands whose nearest product lies at least one binade inside the normal
  // range, even after an output modifier, can only be inexact: no guest rounding mode
  // moves them across a threshold. The FP64 product is exact under the default host FP
  // environment, so this needs neither host exception flags nor the integer path.
  if ((ma >> 23) - 1U < 254U && (mb >> 23) - 1U < 254U) {
    const double product = double(as_float(a)) * double(as_float(b));
    const float nearest = float(product);
    if ((as_bits(nearest) >> 23 & 255U) - 2U < 250U) {
      return (modifiers & RJ_ARITH_ALU_OMOD_MASK) || double(nearest) == product
                 ? 0
                 : RJ_ARITH_EXCEPTION_INEXACT;
    }
  }
  const bool signaling =
      (ma > 0x7f800000U && !(ma & 0x00400000U)) || (mb > 0x7f800000U && !(mb & 0x00400000U));
  if (ma > 0x7f800000U || mb > 0x7f800000U) {
    return signaling ? RJ_ARITH_EXCEPTION_INVALID : 0;
  }
  uint32_t causes = 0;
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    ma = ma < 0x00800000U ? 0 : ma;
    mb = mb < 0x00800000U ? 0 : mb;
  } else if ((ma && ma < 0x00800000U) || (mb && mb < 0x00800000U)) {
    causes = RJ_ARITH_EXCEPTION_INPUT_DENORM;
  }
  if ((!ma && mb == 0x7f800000U) || (!mb && ma == 0x7f800000U)) {
    return causes | RJ_ARITH_EXCEPTION_INVALID;
  }
  if (ma >= 0x7f800000U || mb >= 0x7f800000U || !ma || !mb) {
    return causes;
  }
  // The signed product decides the direction of guest directed rounding.
  a = (a & 0x80000000U) | ma;
  b = (b & 0x80000000U) | mb;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const uint64_t exact = fma_finite_round_odd(a, b, 0);
  const uint32_t result =
      fma_narrow_f32(exact, flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, rounding);
  uint32_t status = rounding_causes<false>(exact, result, rounding);
  const unsigned omod = (modifiers >> 6) & 3;
  if (omod) {
    status &= RJ_ARITH_EXCEPTION_OVERFLOW;
    if (!status && omod != 3) {
      // Scaling by 2 or 4 adds to the exponent of the unbounded-exponent rounding.
      status = rounding_causes<false>(exact + (uint64_t(omod) << 52), 0, rounding) &
               RJ_ARITH_EXCEPTION_OVERFLOW;
    }
  }
  return causes | status;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
