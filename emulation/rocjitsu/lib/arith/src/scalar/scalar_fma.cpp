// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar fused multiply-add under guest FP semantics.
// Owns public entry points and their validation/dispatch policy.

// Fused rounding and tininess detection follow rocjitsu shared/fp_mode.h.
// FP16 uses the existing rocjitsu-derived half_fma_value model. Exact results
// match gfx1201 captures in every MODE setting.

#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/fma_scalar.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/fp_status.h"
#include "scalar/gfx1150_policy.h"

#include <stdint.h>

namespace {

// Evaluate scalar fused multiply-add. Reject invalid controls before changing outputs.
template <bool Half, bool Accumulate, bool Legacy = false, bool Gfx1150 = false>
[[gnu::always_inline]] inline int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a,
                                      uint32_t b, uint32_t c) {
  if (int error = rocjitsu::arith::validate(
          flags, mode,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_ROUND_MASK | (Gfx1150 ? RJ_ARITH_FP_IEEE_MODE : 0))) {
    return error;
  }
  if constexpr (Accumulate) {
    c = *d;
  }
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
    c &= 65535;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf)) {
      a &= sign;
    }
    if (!(b & inf)) {
      b &= sign;
    }
    if (!(c & inf)) {
      c &= sign;
    }
  }
  // Loose nearest-even FP32 uses the host FMA. gfx1250's inherited model also uses it
  // for loose FP32 in any rounding mode, and for FP16, which rounds once to FP32 and
  // then to FP16. The RDNA4 FP16 model rounds, saturates and flushes in every
  // semantics mode.
  if constexpr (!Half || Legacy) {
    if ((Half && Legacy) ||
        (!rocjitsu::arith::is_exact(flags) && (Legacy || !(flags & RJ_ARITH_FP_ROUND_MASK)))) {
      float x = Half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
      float y = Half ? rocjitsu::arith::f16_to_float(uint16_t(b)) : rocjitsu::arith::as_float(b);
      float z = Half ? rocjitsu::arith::f16_to_float(uint16_t(c)) : rocjitsu::arith::as_float(c);
      uint32_t result =
          Half ? rocjitsu::arith::float_to_f16(__builtin_fmaf(x, y, z), flags & RJ_ARITH_FP16_OVFL)
               : rocjitsu::arith::as_bits(__builtin_fmaf(x, y, z));
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        uint32_t magnitude = result & (sign - 1);
        constexpr uint32_t normal = Half ? 0x400 : 0x800000;
        if (magnitude < normal) {
          result &= sign;
        }
        if (magnitude == normal) {
          // TwoSum retains tiny addends when deciding which side of the tininess
          // threshold the exact result occupies. The product is exact in double.
          double product = double(x) * double(y), sum = product + double(z), part = sum - product;
          double residual = (product - (sum - part)) + (double(z) - part);
          double absolute = (result & sign) ? -sum : sum,
                 tail = (result & sign) ? -residual : residual;
          constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
          if (absolute < threshold || (absolute == threshold && tail < 0)) {
            result &= sign;
          }
        }
      }
      *d = result;
      return RJ_ARITH_SUCCESS;
    }
  }
  if constexpr (Half) {
    *d = rocjitsu::arith::half_fma_value<Gfx1150>(uint16_t(a), uint16_t(b), uint16_t(c), 0, flags);
  } else {
    *d = rocjitsu::arith::fma_value_f32(a, b, c, flags, Gfx1150);
  }
  return RJ_ARITH_SUCCESS;
}

// Carry inherited gfx12 Compatibility exception causes alongside the numeric result.
template <bool Half, bool Accumulate, bool Gfx1150 = false>
int run_policy(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t c,
               uint32_t *status) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if constexpr (Gfx1150) {
    if (int error = rocjitsu::arith::prepare_gfx1150_scalar(flags)) {
      return error;
    }
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  const bool report = rocjitsu::arith::reports_fp_status(flags);
  if constexpr (Gfx1150) {
    if (report) {
      return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
    }
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if constexpr (Accumulate) {
    c = *d;
  }
  const uint64_t numeric_flags = rocjitsu::arith::without_fp_status(flags);
  int error = run<Half, false, false, Gfx1150>(numeric_flags, uint32_t(mode), d, a, b, c);
  if (!error && (compatibility || report)) {
    *status |= rocjitsu::arith::scalar_fma_causes<Half, Gfx1150>(numeric_flags, a, b, c, *d);
  }
  return error;
}

// Retain the legacy gfx1250 scalar FMA model without advertising new empirical support.
// FP32 Compatibility applies guest rounding; FP16 Compatibility ignores guest controls.
template <bool Half, bool Accumulate>
int run_gfx1250(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t c) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  flags = rocjitsu::arith::resolve_compatibility(
      flags, Half ? RJ_ARITH_SEMANTICS_LOOSE : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate(flags, mode,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                (Half ? 0 : RJ_ARITH_FP_ROUND_MASK))) {
    return error;
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if constexpr (Half) {
    if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
      flags &= ~(RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                 RJ_ARITH_FP16_OVFL);
    }
  }
  return run<Half, Accumulate, true>(flags, 0, d, a, b, c);
}

} // namespace

// Accumulate fused A*B into D.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmac_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<false, true>(flags, instruction_flags, d, a, b, 0, excp_flag_user);
}

// Accumulate fused A*B into D.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmac_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<true, true>(flags, instruction_flags, d, a, b, 0, excp_flag_user);
}

// Compute fused A*B+literal.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmaak_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t literal, uint32_t *excp_flag_user) {
  return run_policy<false, false>(flags, instruction_flags, d, a, b, literal, excp_flag_user);
}

// Compute fused A*literal+B.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_fmamk_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t literal, uint32_t c, uint32_t *excp_flag_user) {
  return run_policy<false, false>(flags, instruction_flags, d, a, literal, c, excp_flag_user);
}

// Execute gfx1250 scalar fmac_f16; FP status is unused.
int rj_arith_s_fmac_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<true, true>(flags, instruction_flags, d, a, b, 0);
}

// Execute gfx1250 scalar fmac_f32; FP status is unused.
int rj_arith_s_fmac_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<false, true>(flags, instruction_flags, d, a, b, 0);
}

// Execute gfx1250 scalar fmaak_f32; FP status is unused.
int rj_arith_s_fmaak_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t b, uint32_t c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<false, false>(flags, instruction_flags, d, a, b, c);
}

// Execute gfx1250 scalar fmamk_f32; FP status is unused.
int rj_arith_s_fmamk_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t b, uint32_t c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<false, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx1150 scalar fmac with the inherited guest rounding and IEEE policy.
int rj_arith_s_fmac_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t b, uint32_t *fp_status) {
  return run_policy<true, true, true>(flags, instruction_flags, d, a, b, 0, fp_status);
}

// Evaluate gfx1150 scalar fmac with the inherited guest rounding and IEEE policy.
int rj_arith_s_fmac_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t b, uint32_t *fp_status) {
  return run_policy<false, true, true>(flags, instruction_flags, d, a, b, 0, fp_status);
}

// Evaluate gfx1150 scalar fmaak with the inherited guest rounding and IEEE policy.
int rj_arith_s_fmaak_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t b, uint32_t c, uint32_t *fp_status) {
  return run_policy<false, false, true>(flags, instruction_flags, d, a, b, c, fp_status);
}

// Evaluate gfx1150 scalar fmamk with the inherited guest rounding and IEEE policy.
int rj_arith_s_fmamk_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t b, uint32_t c, uint32_t *fp_status) {
  return run_policy<false, false, true>(flags, instruction_flags, d, a, b, c, fp_status);
}
