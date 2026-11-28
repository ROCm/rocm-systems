// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 unary math, integral rounding, and special-value policies.
// Owns public entry points and their validation/dispatch policy.

#include "math/half_unary.h"
#include "common/alu.h"
#include "common/float_encoding.h"
#include "common/sdwa_dispatch.h"
#include "common/validation.h"
#include "math/half_unary_exact.h"
#include "math/unary_impl.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {

// FP16 square root consumes the DPP temporary in AVX2 chunks. Route it at the
// same width so the store and the dependent load use the same granularity.
template <rocjitsu::arith::Unary Op> uint64_t routing_level(uint64_t flags) {
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      return RJ_ARITH_CPU_X86_64_V3;
    }
  }
#endif
  return flags;
}

template <rocjitsu::arith::Unary Op>
constexpr bool supports_exact =
    Op == rocjitsu::arith::Unary::Trunc || Op == rocjitsu::arith::Unary::Ceil ||
    Op == rocjitsu::arith::Unary::Rndne || Op == rocjitsu::arith::Unary::Floor ||
    Op == rocjitsu::arith::Unary::FrexpMant || Op == rocjitsu::arith::Unary::Exp ||
    Op == rocjitsu::arith::Unary::Log || rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op>;

static const uint64_t fp_flags = RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                 RJ_ARITH_FP_DX10_CLAMP_OFF;

// Evaluate FP16 unary math for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false, bool PreserveClampNan = false>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(routing_level<Op>(flags), exec_mask, mode, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return unary_exact<Op, Rdna3, PreserveClampNan>(
                                              flags, exec_mask, uint32_t(mode), d, source);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, fp_flags)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // Actual Loose does not require directed narrowing; keep the shared SIMD route available.
  if ((flags & RJ_ARITH_FP_ROUND_MASK) &&
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_LOOSE) {
    flags &= ~RJ_ARITH_FP_ROUND_MASK;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op>) {
    if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::half_unary_exact_x86_64_v3<Op>(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                                      !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  if constexpr (Op == rocjitsu::arith::Unary::Trunc || Op == rocjitsu::arith::Unary::Ceil ||
                Op == rocjitsu::arith::Unary::Floor || Op == rocjitsu::arith::Unary::Rndne ||
                Op == rocjitsu::arith::Unary::FrexpMant || Op == rocjitsu::arith::Unary::Fract) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if ((Op == rocjitsu::arith::Unary::FrexpMant || Op == rocjitsu::arith::Unary::Fract ||
         !(flags & RJ_ARITH_FP_ROUND_MASK)) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::half_unary_exact_x86_64_v4<Op>(flags | (Rdna3 ? 0 : RJ_ARITH_FP_IEEE_MODE),
                                                      exec_mask, uint32_t(mode), d[0], a[0]);
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if ((Op == rocjitsu::arith::Unary::FrexpMant || Op == rocjitsu::arith::Unary::Fract ||
         !(flags & RJ_ARITH_FP_ROUND_MASK)) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::half_unary_x86_64_v3<Op, true, Op == rocjitsu::arith::Unary::Fract>(
          flags | (Rdna3 ? 0 : RJ_ARITH_FP_IEEE_MODE), exec_mask, uint32_t(mode), d[0], a[0]);
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  unsigned a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::half_unary_exact_value<Op, Rdna3, PreserveClampNan>(
        flags, uint32_t(mode), uint16_t(a[0][lane] >> a_shift));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(65535U << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 unary math for the selected lanes. Reject invalid controls before changing
// outputs.
// FRACT has one model and no exact kernel; entry points checked against their architecture's
// captures pass AcceptExact so strict exact runs this path.
template <rocjitsu::arith::Unary Op, bool GuestDenormals = true, bool PreserveClampNan = false,
          bool AcceptExact = false>
[[gnu::noinline]] int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (modifiers >> 32) {
    return rocjitsu::arith::execute_dpp(
        routing_level<Op>(flags), exec_mask, modifiers, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary<Op, GuestDenormals, PreserveClampNan, AcceptExact>(
              flags, exec_mask, uint32_t(modifiers), d, source);
        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, modifiers & ~known, fp_flags)) {
    return error;
  }
  if (!AcceptExact &&
      (flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // TRANS arithmetic and its overflow use fixed rounding, independently of guest ROUND.
  if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> || Op == rocjitsu::arith::Unary::Exp ||
                Op == rocjitsu::arith::Unary::Log) {
    flags &= ~RJ_ARITH_FP_ROUND_MASK;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  // Loose may use host nearest rounding, including when guest ROUND is directed.
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_LOOSE) {
    flags &= ~RJ_ARITH_FP_ROUND_MASK;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (Op == rocjitsu::arith::Unary::Fract) {
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      // The exact FRACT kernel is also the cheapest Loose implementation.
      rocjitsu::arith::half_unary_exact_x86_64_v4<Op>(flags, exec_mask, uint32_t(modifiers), d[0],
                                                      a[0]);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> || Op == rocjitsu::arith::Unary::Exp ||
                Op == rocjitsu::arith::Unary::Log) {
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::half_unary_loose_x86_64_v4<Op>(flags, exec_mask, uint32_t(modifiers), d[0],
                                                      a[0]);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (Op == rocjitsu::arith::Unary::Fract) {
    if ((flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::half_unary_x86_64_v3<Op, GuestDenormals, true>(
          flags, exec_mask, uint32_t(modifiers), d[0], a[0]);
      return RJ_ARITH_SUCCESS;
    }
  }
  if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_unary_x86_64_v3<Op, GuestDenormals>(flags, exec_mask, modifiers, d[0],
                                                              a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if constexpr (supports_exact<Op> && !rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> &&
                Op != rocjitsu::arith::Unary::Exp && Op != rocjitsu::arith::Unary::Log) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return unary_exact<Op, false, true>(flags, exec_mask, modifiers, d, a);
    }
    return unary_exact<Op>(flags, exec_mask, modifiers, d, a);
  }
  int a_shift = modifiers & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int d_shift = modifiers & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t input = uint16_t(a[0][lane] >> a_shift);
    if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> ||
                  Op == rocjitsu::arith::Unary::Fract || Op == rocjitsu::arith::Unary::Exp ||
                  Op == rocjitsu::arith::Unary::Log) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
          (input & 0x7fff) < 0x400) {
        input &= 0x8000;
      }
    }
    float x;
    if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
      if (modifiers & RJ_ARITH_ALU_ABS_A) {
        input &= 0x7fff;
      }
      if (modifiers & RJ_ARITH_ALU_NEG_A) {
        input ^= 0x8000;
      }
      const uint16_t magnitude = input & 0x7fff;
      if (magnitude >= 0x7c00 ||
          (Op == rocjitsu::arith::Unary::Log && ((input & 0x8000) || !magnitude))) {
        // Reuse raw special-value handling rather than convert NaNs/infinities repeatedly.
        result[lane] = rocjitsu::arith::half_unary_exact_value<Op, false, PreserveClampNan>(
            flags, uint32_t(modifiers & ~(RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A)), input);
        continue;
      }
      x = rocjitsu::arith::f16_to_float(input);
      if constexpr (Op == rocjitsu::arith::Unary::Exp) {
        // Special values were handled above; bound the native finite exponential.
        x = rocjitsu::arith::clamp(x, -64.0f, 64.0f);
      }
    } else {
      x = rocjitsu::arith::alu_input(rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(input)),
                                     modifiers);
    }
    float value = rocjitsu::arith::unary_value<Op>(x);
    if constexpr (Op == rocjitsu::arith::Unary::Rcp || Op == rocjitsu::arith::Unary::Rsq) {
      if ((flags & RJ_ARITH_FP16_OVFL) && x == 0) {
        value = __builtin_copysignf(65504.0f, value);
      }
    }
    if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op>) {
      // CLAMP removes the overflow distinction; otherwise retain explicit root saturation.
      if ((modifiers & RJ_ARITH_ALU_OMOD_MASK) && !(modifiers & RJ_ARITH_ALU_CLAMP)) {
        value = rocjitsu::arith::f16_to_float(
            rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL));
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        // Loose needs no intermediate half rounding before final scaling/narrowing.
        if (__builtin_fabsf(value) < 0x1p-14f) {
          value = __builtin_copysignf(0.0f, value);
        }
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Fract) {
      // Round the exact half-input subtraction before OMOD, using guest MODE.
      if (__builtin_isfinite(x)) {
        uint16_t half = rocjitsu::arith::half_fma_narrow(
            rocjitsu::arith::bit_cast<uint64_t>(double(x) - __builtin_floor(double(x))),
            flags & RJ_ARITH_FP16_OVFL, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
        value = rocjitsu::arith::f16_to_float(half);
      }
      // Keep a fraction strictly below one after FP16 rounding.
      if (value > 0x1.ffcp-1f) {
        value = 0x1.ffcp-1f;
      }
    }
    if constexpr (rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> ||
                  Op == rocjitsu::arith::Unary::Fract) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
          __builtin_fabs(value) < 0x1p-14f) {
        value = __builtin_copysignf(0.0f, value);
      }
    }
    uint32_t output_mode = modifiers;
    if constexpr (PreserveClampNan) {
      if ((rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000) {
        output_mode &= ~RJ_ARITH_ALU_CLAMP;
      }
    }
    float output = rocjitsu::arith::alu_output_f16(value, output_mode);
    result[lane] =
        (flags & RJ_ARITH_FP_ROUND_MASK) && __builtin_isfinite(output)
            ? rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(double(output)),
                                               flags & RJ_ARITH_FP16_OVFL,
                                               unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24))
            : rocjitsu::arith::float_to_f16(output, flags & RJ_ARITH_FP16_OVFL);
    if constexpr (Op == rocjitsu::arith::Unary::Fract) {
      uint16_t raw = uint16_t(a[0][lane] >> a_shift);
      if (modifiers & RJ_ARITH_ALU_ABS_A) {
        raw &= 0x7fff;
      }
      if (modifiers & RJ_ARITH_ALU_NEG_A) {
        raw ^= 0x8000;
      }
      if (!(flags & RJ_ARITH_FP_IEEE_MODE) && (raw & 0x7fff) > 0x7c00 &&
          (!(modifiers & RJ_ARITH_ALU_CLAMP) || PreserveClampNan)) {
        result[lane] = raw;
      }
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}
// Evaluate FP16 unary math for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false, bool QuietNan = false,
          bool AcceptExact = false, bool OlderProfile = false>
[[gnu::always_inline]] inline int run(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      [[maybe_unused]] uint32_t *state) {
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if constexpr (OlderProfile) {
    // Reused older-target arithmetic is not empirically qualified for strict modes.
    if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
      return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
    }
  }
  if constexpr (!Rdna3 && !OlderProfile) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  if constexpr (Rdna3 || OlderProfile) {
    if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
    }
  }
  // Resolve guest output controls before applying the CDNA NaN policy.
  if constexpr (QuietNan) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  if constexpr (supports_exact<Op> || (OlderProfile && Op == rocjitsu::arith::Unary::Fract)) {
    // Integral results and normalized mantissas have one cheap model for both semantics.
    constexpr bool shared_numeric =
        rocjitsu::arith::unary_rounds_f32<Op> || Op == rocjitsu::arith::Unary::FrexpMant;
    if (shared_numeric || QuietNan || rocjitsu::arith::is_exact(flags) || OlderProfile) {
      constexpr bool rdna3_nan_policy = Rdna3;
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        return unary_exact<Op, rdna3_nan_policy, true>(flags, exec_mask, mode, d, a);
      }
      return unary_exact<Op, rdna3_nan_policy>(flags, exec_mask, mode, d, a);
    }
  }
  if constexpr (!Rdna3) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
    if (!(flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        return unary<Op, false, true, AcceptExact>(flags, exec_mask, mode, d, a);
      }
      return unary<Op, false, false, AcceptExact>(flags, exec_mask, mode, d, a);
    }
  }
  if constexpr (!supports_exact<Op> || rocjitsu::arith::half_unary_rcp_rsq_sqrt<Op> ||
                Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return unary<Op, true, true, AcceptExact>(flags, exec_mask, mode, d, a);
    }
  }
  return unary<Op, true, false, AcceptExact>(flags, exec_mask, mode, d, a);
}

// Evaluate FP16 unary math for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false, bool QuietNan = false,
          bool AcceptExact = false, bool OlderProfile = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, [[maybe_unused]] uint32_t *state) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Op, Rdna3, QuietNan, AcceptExact, OlderProfile>(flags, 0, mode, nullptr,
                                                                      nullptr, state)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = run<Op, Rdna3, QuietNan, AcceptExact, OlderProfile>(
            flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags, &dest, &source,
            state)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Resolve older TRANS semantics and fixed rounding without qualifying new targets.
// Reject strict requests before output access; rewrite only this call's local flags.
int prepare_older_half_trans_flags(uint64_t &flags) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // Compatibility uses the inherited arithmetic; modes 0/1 retain Loose SIMD.
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  // TRANS and its OMOD overflow ignore guest ROUND; do not disable SIMD for it.
  flags &= ~RJ_ARITH_FP_ROUND_MASK;
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a,
                                                 trapsts);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a,
                                           excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Floor>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                 trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a,
                                           excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a,
                                          excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a,
                                          excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Exp>(flags, exec_mask, instruction_flags, d, a,
                                          excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Log>(flags, exec_mask, instruction_flags, d, a,
                                          excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Fract, true, false, true>(flags, exec_mask, instruction_flags,
                                                               d, a, trapsts);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Fract, false, false, true>(flags, exec_mask, instruction_flags,
                                                                d, a, excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags, d, a,
                                                      trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::FrexpMant>(flags, exec_mask, instruction_flags, d, a,
                                                excp_flag_user);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a,
                                                          trapsts);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a,
                                                    excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                          trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a,
                                                    excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, true, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags,
                                                               d, a, trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant>(flags, exec_mask, instruction_flags, d, a,
                                                         excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Evaluate trunc with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_trunc_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Trunc, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate trunc with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_trunc_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Trunc, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate ceil with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_ceil_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Ceil, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate ceil with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_ceil_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Ceil, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate floor with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_floor_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Floor, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate floor with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_floor_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Floor, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate rndne with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_rndne_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Rndne, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate rndne with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_rndne_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Rndne, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate frexp_mant with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_frexp_mant_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::FrexpMant, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Evaluate frexp_mant with the inherited CDNA NaN and guest-mode policy.
int rj_arith_v_frexp_mant_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::FrexpMant, true, true>(
            rocjitsu::arith::sdwa_fixed_round_flags(flags, instruction_flags), exec_mask, core_mode,
            d, a, trapsts);
      });
}

// Fractional part with inherited CDNA guest rounding and NaN policy.
int rj_arith_v_fract_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  // Keep inherited guest rounding distinguishable from actual Loose's native narrowing.
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Fract, true, true>(flags, exec_mask, core_mode, d, a,
                                                              trapsts);
      });
}

// Fractional part with inherited CDNA guest rounding and NaN policy.
int rj_arith_v_fract_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  // Keep inherited guest rounding distinguishable from actual Loose's native narrowing.
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_output<rocjitsu::arith::SdwaOutput::F16>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Fract, true, true>(flags, exec_mask, core_mode,
                                                                       d, a, trapsts);
      });
}

// Older-profile TRUNC keeps legacy NaN quieting and guest output controls.
int rj_arith_v_trunc_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Trunc, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile CEIL keeps legacy NaN quieting and guest output controls.
int rj_arith_v_ceil_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Ceil, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile RNDNE keeps legacy NaN quieting and guest output controls.
int rj_arith_v_rndne_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rndne, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile FLOOR keeps legacy NaN quieting and guest output controls.
int rj_arith_v_floor_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Floor, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile TRUNC keeps legacy NaN quieting and guest output controls.
int rj_arith_v_trunc_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile CEIL keeps legacy NaN quieting and guest output controls.
int rj_arith_v_ceil_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile RNDNE keeps legacy NaN quieting and guest output controls.
int rj_arith_v_rndne_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile FLOOR keeps legacy NaN quieting and guest output controls.
int rj_arith_v_floor_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile FREXP mantissas keep legacy NaN quieting and output controls.
int rj_arith_v_frexp_mant_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::FrexpMant, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile FREXP mantissas keep legacy NaN quieting and output controls.
int rj_arith_v_frexp_mant_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile RCP reuses the inherited mapping without strict target qualification.
int rj_arith_v_rcp_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, core_mode, d, a, trapsts);
      });
}

// Older-profile RCP reuses the inherited mapping without strict target qualification.
int rj_arith_v_rcp_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, core_mode, d, a,
                                                               trapsts);
      });
}

// Older-profile RSQ reuses the inherited mapping without strict target qualification.
int rj_arith_v_rsq_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, core_mode, d, a, trapsts);
      });
}

// Older-profile RSQ reuses the inherited mapping without strict target qualification.
int rj_arith_v_rsq_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, core_mode, d, a,
                                                               trapsts);
      });
}

// Older-profile SQRT reuses the inherited mapping without strict target qualification.
int rj_arith_v_sqrt_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, core_mode, d, a, trapsts);
      });
}

// Older-profile SQRT reuses the inherited mapping without strict target qualification.
int rj_arith_v_sqrt_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, core_mode, d, a,
                                                                trapsts);
      });
}

// Older-profile EXP retains the existing half model in Compatibility and SIMD in Loose.
int rj_arith_v_exp_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, core_mode, d, a, trapsts);
      });
}

// Older-profile EXP retains the existing half model in Compatibility and SIMD in Loose.
int rj_arith_v_exp_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, core_mode, d, a,
                                                               trapsts);
      });
}

// Older-profile LOG retains the existing half model in Compatibility and SIMD in Loose.
int rj_arith_v_log_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return run<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, core_mode, d, a, trapsts);
      });
}

// Older-profile LOG retains the existing half model in Compatibility and SIMD in Loose.
int rj_arith_v_log_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (int error = prepare_older_half_trans_flags(flags)) {
    return error;
  }
  return rocjitsu::arith::with_sdwa_half_finalization(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return unary_wave64<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, core_mode, d, a,
                                                               trapsts);
      });
}

// Older-profile FRACT preserves guest rounding, denormals and rounded-result output controls.
int rj_arith_v_fract_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Fract, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Older-profile FRACT preserves guest rounding, denormals and rounded-result output controls.
int rj_arith_v_fract_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, false, false, false, true>(
      flags, exec_mask, instruction_flags, d, a, trapsts);
}
