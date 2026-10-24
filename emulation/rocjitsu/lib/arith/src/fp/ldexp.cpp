// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// Owns public entry points and their validation/dispatch policy.

#include "fp/ldexp.h"
#include "common/alu.h"
#include "common/internal.h"
#include "fp/float_output.h"
#include "fp/fp64.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace {

// Adapted from rocjitsu shared/fp_mode.h::ldexp and division.h::div_round.
// The significand fits in its native word; all rounding is integer-only.
template <int Width>
typename rocjitsu::arith::IeeeFormat<Width>::Bits
ldexp_bits(typename rocjitsu::arith::IeeeFormat<Width>::Bits bits, int32_t adjustment,
           uint64_t flags) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using T = typename F::Bits;
  const T implicit = T(1) << F::fraction;
  const T sign = bits & F::sign, magnitude = bits & (F::sign - 1);
  if (magnitude >= F::infinity) {
    return bits | (magnitude > F::infinity ? F::quiet : 0);
  }
  int exponent = int(magnitude >> F::fraction);
  T significand = magnitude & (implicit - 1);
  if (!magnitude || (!exponent && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS))) {
    return sign;
  }
  if (exponent) {
    significand |= implicit;
  } else {
    exponent = 1;
    while (significand < implicit) {
      significand <<= 1;
      --exponent;
    }
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool outward = rounding == (sign ? 2u : 1u);
  const T overflow = sign | (rounding == 0 || outward ? F::infinity : F::infinity - 1);
  adjustment = adjustment < -4096 ? -4096 : adjustment > 4096 ? 4096 : adjustment;
  exponent += adjustment;
  if (exponent >= 2 * F::bias + 1) {
    return overflow;
  }
  if (exponent > 0) {
    return sign | (T(exponent) << F::fraction) | (significand & (implicit - 1));
  }
  const unsigned shift = unsigned(1 - exponent);
  T kept = shift >= Width ? 0 : significand >> shift;
  const bool guard = shift <= Width && ((significand >> (shift - 1)) & 1);
  const bool sticky =
      shift > Width ? significand != 0 : shift > 1 && (significand << (Width + 1 - shift)) != 0;
  if ((rounding == 0 && guard && (sticky || (kept & 1))) || (outward && (guard || sticky))) {
    ++kept;
  }
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    kept = 0;
  }
  return sign | kept;
}

// Scale floating-point encodings by powers of two with guest rounding and output modifiers.
template <bool Fp64>
int ldexp_rounded(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b) {
  constexpr unsigned width = Fp64 ? 64 : 32;
  using F = rocjitsu::arith::IeeeFormat<width>;
  using T = typename F::Bits;
  uint32_t result[Fp64 ? 2 : 1][32];
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; ++lane) {
    T input = a[0][lane];
    if constexpr (Fp64) {
      input |= T(a[1][lane]) << 32;
    }
    if (mode & RJ_ARITH_ALU_ABS_A) {
      input &= F::sign - 1;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      input ^= F::sign;
    }
    int32_t adjustment = std::bit_cast<int32_t>(b[0][lane]);
    // Tininess is tested before rounding; active OMOD also flushes that domain.
    T value = ldexp_bits<width>(
        input, adjustment,
        flags | ((mode & RJ_ARITH_ALU_OMOD_HALF) ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0));
    const bool outward = rounding == (value & F::sign ? 2u : 1u);
    const T overflow = rounding == 0 || outward ? F::infinity : F::infinity - 1;
    const bool preserve_nan =
        (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (value & (F::sign - 1)) > F::infinity;
    value = rocjitsu::arith::float_output<width>(
        value, preserve_nan ? mode & ~RJ_ARITH_ALU_CLAMP : mode, overflow);
    result[0][lane] = uint32_t(value);
    if constexpr (Fp64) {
      result[1][lane] = uint32_t(value >> 32);
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate LDEXP with guest rounding/denormal controls, preserving inactive destinations.
template <bool Fp64, bool Rdna3>
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Scale FP32/FP64 by powers of two for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Fp64, bool GuestDenormals = false, bool Rdna3 = false>
int ldexp(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known, true,
          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_MASK |
              (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_LOOSE ||
      (flags & RJ_ARITH_FP_ROUND_MASK)) {
    return ldexp_rounded<Fp64>(flags, exec_mask, mode, d, a, b);
  }
  if constexpr (!GuestDenormals) {
    if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                 RJ_ARITH_FP_DX10_CLAMP_OFF)) {
      return ldexp_guest<Fp64, Rdna3>(flags, exec_mask, mode, d, a, b);
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    if constexpr (GuestDenormals) {
      rocjitsu::arith::ldexp_guest_x86_64_v4(Fp64, flags, exec_mask, mode, d, a, b[0]);
    } else {
      rocjitsu::arith::ldexp_x86_64_v4(Fp64, exec_mask, mode, d, a, b[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    if constexpr (GuestDenormals) {
      rocjitsu::arith::ldexp_guest_x86_64_v3(Fp64, flags, exec_mask, mode, d, a, b[0]);
    } else {
      rocjitsu::arith::ldexp_x86_64_v3(Fp64, exec_mask, mode, d, a, b[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Fp64 ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    int32_t exponent = std::bit_cast<int32_t>(b[0][lane]);
    if constexpr (Fp64) {
      double input = rocjitsu::arith::fp64_input(a, lane, mode);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        input = rocjitsu::arith::fp64_flush_denorm(input);
      }
      double value = std::ldexp(input, exponent);
      if (((mode & RJ_ARITH_ALU_OMOD_HALF) ||
           (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) &&
          std::abs(value) == 0x1p-1022) {
        // A tiny exact result can round up to minimum normal. Detect this
        // from the source exponent without relying on wider host FP types.
        int source_exponent;
        std::frexp(input, &source_exponent);
        if (int64_t(source_exponent) + exponent <= -1022) {
          value = mode & RJ_ARITH_ALU_OMOD_HALF ? 0.0 : std::copysign(0.0, value);
        }
      }
      const bool preserve_nan =
          (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
          (rocjitsu::arith::double_bits(value) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
      value = rocjitsu::arith::fp64_output(value, preserve_nan ? mode & ~RJ_ARITH_ALU_CLAMP : mode);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        value = rocjitsu::arith::fp64_flush_denorm(value);
      }
      uint64_t bits = rocjitsu::arith::double_bits(value);
      result[0][lane] = uint32_t(bits);
      result[1][lane] = uint32_t(bits >> 32);
    } else {
      float input = rocjitsu::arith::alu_input(a[0][lane], mode);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        input = rocjitsu::arith::flush_denorm_f32(input);
      }
      float value;
      if ((mode & RJ_ARITH_ALU_OMOD_HALF) ||
          (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
        // OMOD/output flushing tests tininess before FP32 rounding, including values
        // that would round upward to minimum normal.
        double wide = std::ldexp(double(input), exponent);
        value = std::abs(wide) < 0x1p-126
                    ? (mode & RJ_ARITH_ALU_OMOD_HALF ? 0.0f : std::copysign(0.0f, input))
                    : float(wide);
      } else {
        value = std::ldexp(input, exponent);
      }
      // DX10_CLAMP-off preserves NaNs rather than clamping them to zero.
      const bool preserve_nan = (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
                                (rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000;
      value =
          rocjitsu::arith::alu_output_f32(value, preserve_nan ? mode & ~RJ_ARITH_ALU_CLAMP : mode);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        value = rocjitsu::arith::flush_denorm_f32(value);
      }
      result[0][lane] = rocjitsu::arith::as_bits(value);
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Keep guest denormal handling out of the default scalar loop. Inlining it
// allows the compiler to merge both loops and reintroduce per-lane flag checks.
template <bool Fp64, bool Rdna3>
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return ldexp<Fp64, true, Rdna3>(flags, exec_mask, mode, d, a, b);
}

// Scale FP32/FP64 by powers of two for Wave64. Reject invalid controls before changing outputs.
template <bool Fp64, bool Rdna3 = false>
int ldexp_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  constexpr auto instruction =
      Fp64 ? (Rdna3 ? rj_arith_v_ldexp_f64_w32 : rj_arith_v_ldexp_f64_w32_gfx12)
           : (Rdna3 ? rj_arith_v_ldexp_f32_w32 : rj_arith_v_ldexp_f32_w32_gfx12);

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Fp64 ? 2 : 1];
    const uint32_t *source[Fp64 ? 2 : 1];
    for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
      dest[reg] = d[reg] + half * 32;
      source[reg] = a[reg] + half * 32;
    }
    // The signed integer exponent occupies one VGPR even for FP64.
    const uint32_t *exponent = b[0] + half * 32;
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest, source,
                                &exponent, excp_flag_user)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return ldexp<false>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }
  return ldexp<false>(flags, exec_mask, mode, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<true>(flags, exec_mask, mode, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ldexp_wave64<false>(flags, exec_mask, mode, d, a, b, excp_flag_user);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ldexp_wave64<true>(flags, exec_mask, mode, d, a, b, excp_flag_user);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return ldexp<false, false, true>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }
  return ldexp<false, false, true>(flags, exec_mask, mode, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return ldexp_wave64<false, true>(flags, exec_mask, mode, d, a, b, trapsts);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<true, false, true>(flags, exec_mask, mode, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return ldexp_wave64<true, true>(flags, exec_mask, mode, d, a, b, trapsts);
}
