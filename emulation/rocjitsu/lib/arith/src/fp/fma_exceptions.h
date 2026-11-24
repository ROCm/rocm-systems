// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest FP32 add/sub/FMA exception classification from operands and controls.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/alu.h"
#include "common/bits.h"
#include "common/validation.h"
#include "fp/binary.h"
#include "fp/fma_integer.h"
#include "fp/fma_scalar.h"
#include "fp/rounding_status.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Classify one gfx12 FP32 FMA (a * b + c) lane.
// Fitted to gfx1201 EXCP_FLAG_USER captures (R9700, Nov 18 2026) of V_FMA_F32 and
// V_FMAC_F32 across all 84 MODE settings, OMOD and CLAMP (688K results each, 0 mismatches):
// CLAMP suppresses every cause. Signaling NaNs and invalid products (0 * inf, with zero
// judged after input flushing) raise INVALID and suppress INPUT_DENORM even when C is a
// quiet NaN; opposite infinities raise INVALID and suppress INPUT_DENORM. Overflow and
// tininess follow rounding_causes; OMOD suppresses UNDERFLOW and INEXACT and judges
// OVERFLOW before and after scaling under the guest rounding mode.
inline uint32_t fma_exceptions_f32(uint32_t a, uint32_t b, uint32_t c, uint64_t flags,
                                   uint32_t modifiers) {
  if (modifiers & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }
  a = alu_input_bits(a, modifiers);
  b = alu_input_bits(b, modifiers >> 1);
  c = alu_input_bits(c, modifiers >> 2);
  uint32_t ma = a & 0x7fffffffU, mb = b & 0x7fffffffU, mc = c & 0x7fffffffU;
  const bool a_nan = ma > 0x7f800000U, b_nan = mb > 0x7f800000U, c_nan = mc > 0x7f800000U;
  const bool flush_in = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (a_nan || b_nan || c_nan) {
    const bool signaling = (a_nan && !(ma & 0x00400000U)) || (b_nan && !(mb & 0x00400000U)) ||
                           (c_nan && !(mc & 0x00400000U));
    if (signaling) {
      return RJ_ARITH_EXCEPTION_INVALID;
    }
    if (!a_nan && !b_nan) {
      const uint32_t fa = flush_in && ma < 0x00800000U ? 0 : ma;
      const uint32_t fb = flush_in && mb < 0x00800000U ? 0 : mb;
      if ((!fa && fb == 0x7f800000U) || (!fb && fa == 0x7f800000U)) {
        return RJ_ARITH_EXCEPTION_INVALID;
      }
    }
    return 0;
  }
  uint32_t causes = 0;
  if (flush_in) {
    ma = ma < 0x00800000U ? 0 : ma;
    mb = mb < 0x00800000U ? 0 : mb;
    mc = mc < 0x00800000U ? 0 : mc;
  } else if ((ma && ma < 0x00800000U) || (mb && mb < 0x00800000U) || (mc && mc < 0x00800000U)) {
    causes = RJ_ARITH_EXCEPTION_INPUT_DENORM;
  }
  const bool a_inf = ma == 0x7f800000U, b_inf = mb == 0x7f800000U, c_inf = mc == 0x7f800000U;
  if ((a_inf && !mb) || (b_inf && !ma)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  if (a_inf || b_inf) {
    if (c_inf && ((a ^ b ^ c) & 0x80000000U)) {
      return RJ_ARITH_EXCEPTION_INVALID;
    }
    return causes;
  }
  if (c_inf) {
    return causes;
  }
  a = (a & 0x80000000U) | ma;
  b = (b & 0x80000000U) | mb;
  c = (c & 0x80000000U) | mc;
  const uint64_t exact = fma_finite_round_odd(a, b, c);
  if (!(exact & 0x7fffffffffffffffULL)) {
    return causes;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned omod = (modifiers >> 6) & 3;
  if (omod) {
    uint32_t status = rounding_causes<false>(exact, 0, rounding) & RJ_ARITH_EXCEPTION_OVERFLOW;
    if (!status && omod != 3) {
      status = rounding_causes<false>(exact + (uint64_t(omod) << 52), 0, rounding) &
               RJ_ARITH_EXCEPTION_OVERFLOW;
    }
    return causes | status;
  }
  const uint32_t result =
      fma_narrow_f32(exact, flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, rounding);
  return causes | rounding_causes<false>(exact, result, rounding);
}

// Classify one gfx12 FP32 ADD/SUB/SUBREV lane.
// Fitted to gfx1201 EXCP_FLAG_USER captures (R9700, Nov 18 2026) of V_ADD_F32, V_SUB_F32
// and V_ADD_F32 OMOD across all 84 MODE settings (0 mismatches).
template <Binary Op>
inline uint32_t add_exceptions_f32(uint32_t a, uint32_t b, uint64_t flags, uint32_t modifiers) {
  static_assert(Op == Binary::Add || Op == Binary::Sub || Op == Binary::Subrev);
  if (modifiers & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }
  a = alu_input_bits(a, modifiers);
  b = alu_input_bits(b, modifiers >> 1);
  if constexpr (Op == Binary::Sub) {
    b ^= 0x80000000U;
  }
  if constexpr (Op == Binary::Subrev) {
    a ^= 0x80000000U;
  }
  uint32_t ma = a & 0x7fffffffU, mb = b & 0x7fffffffU;
  // Normal operands within 28 binades have an exact FP64 sum; when the nearest
  // FP32 result stays at least one binade inside normal range, only INEXACT is possible.
  const uint32_t ea = ma >> 23, eb = mb >> 23;
  if (ea - 1U < 254U && eb - 1U < 254U && (ea >= eb ? ea - eb : eb - ea) <= 28U) {
    const double sum = double(as_float(a)) + double(as_float(b));
    if (sum == 0.0) {
      return 0;
    }
    const float nearest = float(sum);
    if ((as_bits(nearest) >> 23 & 255U) - 2U < 250U) {
      return (modifiers & RJ_ARITH_ALU_OMOD_MASK) || double(nearest) == sum
                 ? 0
                 : RJ_ARITH_EXCEPTION_INEXACT;
    }
  }
  const bool a_nan = ma > 0x7f800000U, b_nan = mb > 0x7f800000U;
  if (a_nan || b_nan) {
    return (a_nan && !(ma & 0x00400000U)) || (b_nan && !(mb & 0x00400000U))
               ? RJ_ARITH_EXCEPTION_INVALID
               : 0;
  }
  uint32_t causes = 0;
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    ma = ma < 0x00800000U ? 0 : ma;
    mb = mb < 0x00800000U ? 0 : mb;
  } else if ((ma && ma < 0x00800000U) || (mb && mb < 0x00800000U)) {
    causes = RJ_ARITH_EXCEPTION_INPUT_DENORM;
  }
  const bool a_inf = ma == 0x7f800000U, b_inf = mb == 0x7f800000U;
  if (a_inf && b_inf && ((a ^ b) & 0x80000000U)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  if (a_inf || b_inf) {
    return causes;
  }
  a = (a & 0x80000000U) | ma;
  b = (b & 0x80000000U) | mb;
  const uint64_t exact = fma_finite_round_odd(a, 0x3f800000U, b);
  if (!(exact & 0x7fffffffffffffffULL)) {
    return causes;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned omod = (modifiers >> 6) & 3;
  if (omod) {
    uint32_t status = rounding_causes<false>(exact, 0, rounding) & RJ_ARITH_EXCEPTION_OVERFLOW;
    if (!status && omod != 3) {
      status = rounding_causes<false>(exact + (uint64_t(omod) << 52), 0, rounding) &
               RJ_ARITH_EXCEPTION_OVERFLOW;
    }
    return causes | status;
  }
  const uint32_t result =
      fma_narrow_f32(exact, flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, rounding);
  return causes | rounding_causes<false>(exact, result, rounding);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
