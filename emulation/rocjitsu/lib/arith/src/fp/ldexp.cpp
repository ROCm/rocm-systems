// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// Owns public entry points and their validation/dispatch policy.

#include "fp/ldexp.h"
#include "common/alu.h"
#include "common/bits.h"
#include "common/validation.h"
#include "fp/float_output.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <stdint.h>

namespace {

// Adapted from rocjitsu shared/fp_mode.h::ldexp and division.h::div_round.
// The significand fits in its native word; all rounding is integer-only.
template <int Width>
typename rocjitsu::arith::IeeeFormat<Width>::Bits
ldexp_bits(typename rocjitsu::arith::IeeeFormat<Width>::Bits bits, int32_t adjustment,
           uint64_t flags, bool quiet_snan) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using T = typename F::Bits;
  const T implicit = T(1) << F::fraction;
  const T sign = bits & F::sign, magnitude = bits & (F::sign - 1);
  if (magnitude >= F::infinity) {
    return bits | (magnitude > F::infinity && quiet_snan ? F::quiet : 0);
  }
  int exponent = int(magnitude >> F::fraction);
  T significand = magnitude & (implicit - 1);
  if (!magnitude || (!exponent && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS))) {
    return sign;
  }
  if (exponent) {
    significand |= implicit;
  } else {
    // Move the highest fraction bit into the implicit-one position in one step.
    const int shift =
        rocjitsu::arith::countl_zero(significand) - (int(sizeof(T) * 8) - 1 - F::fraction);
    significand <<= shift;
    exponent = 1 - shift;
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

// Apply the inherited separate OMOD/CLAMP stage to an already scaled encoding.
template <int Width>
[[gnu::always_inline]] inline typename rocjitsu::arith::IeeeFormat<Width>::Bits
ldexp_output(typename rocjitsu::arith::IeeeFormat<Width>::Bits value, uint64_t flags,
             uint32_t instruction_flags) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using T = typename F::Bits;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool outward = rounding == (value & F::sign ? 2u : 1u);
  const T overflow = rounding == 0 || outward ? F::infinity : F::infinity - 1;
  const bool preserve_nan =
      (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (value & (F::sign - 1)) > F::infinity;
  return rocjitsu::arith::float_output<Width>(
      value, preserve_nan ? instruction_flags & ~RJ_ARITH_ALU_CLAMP : instruction_flags, overflow);
}

// Apply source controls, inherited LDEXP rounding and output controls to one lane.
template <int Width>
auto ldexp_value(typename rocjitsu::arith::IeeeFormat<Width>::Bits input, int32_t adjustment,
                 uint64_t flags, uint32_t instruction_flags, bool quiet_snan) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using T = typename F::Bits;
  if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
    input &= F::sign - 1;
  }
  if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
    input ^= F::sign;
  }
  // Tininess is tested before rounding; active OMOD also flushes that domain.
  T value = ldexp_bits<Width>(
      input, adjustment,
      flags |
          ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0),
      quiet_snan);
  return ldexp_output<Width>(value, flags, instruction_flags);
}

// Scale floating-point encodings by powers of two with guest rounding and output modifiers.
template <bool Fp64>
int ldexp_rounded(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  bool quiet_snan) {
  constexpr unsigned width = Fp64 ? 64 : 32;
  using T = typename rocjitsu::arith::IeeeFormat<width>::Bits;
  uint32_t result[Fp64 ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    T input = a[0][lane];
    if constexpr (Fp64) {
      input |= T(a[1][lane]) << 32;
    }
    const int32_t adjustment = rocjitsu::arith::bit_cast<int32_t>(b[0][lane]);
    const T value = ldexp_value<width>(input, adjustment, flags, instruction_flags, quiet_snan);
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

#if defined(RJ_ARITH_HAVE_X86_64_V3) || defined(RJ_ARITH_HAVE_X86_64_V4)
// Stage ordinary scales in SIMD, repair only exceptional lanes with the same
// integer model, then publish both destination registers with masked stores.
int ldexp_f64_simd(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                   bool quiet_snan) {
  uint32_t result[2][32];
  uint32_t scalar_lanes = ~0U;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    scalar_lanes = rocjitsu::arith::ldexp_f64_normal_x86_64_v4(flags, instruction_flags, quiet_snan,
                                                               result, a, b[0]);
  } else
#endif
  {
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    scalar_lanes = rocjitsu::arith::ldexp_f64_normal_x86_64_v3(flags, instruction_flags, quiet_snan,
                                                               result, a, b[0]);
#endif
  }
  if (scalar_lanes) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((scalar_lanes >> lane) & 1) {
        // The SIMD stage retained the modified input for unresolved lanes.
        const uint64_t input = result[0][lane] | (uint64_t(result[1][lane]) << 32);
        const uint64_t value = ldexp_bits<64>(input, rocjitsu::arith::bit_cast<int32_t>(b[0][lane]),
                                              flags | ((instruction_flags & RJ_ARITH_ALU_OMOD_MASK)
                                                           ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS
                                                           : 0),
                                              quiet_snan);
        const uint64_t output = ldexp_output<64>(value, flags, instruction_flags);
        result[0][lane] = uint32_t(output);
        result[1][lane] = uint32_t(output >> 32);
      }
    }
  }
  if (instruction_flags & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP)) {
    for (int lane = 0; lane < 32; ++lane) {
      if (!((scalar_lanes >> lane) & 1)) {
        const uint64_t bits = result[0][lane] | (uint64_t(result[1][lane]) << 32);
        const uint64_t value = ldexp_output<64>(bits, flags, instruction_flags);
        result[0][lane] = uint32_t(value);
        result[1][lane] = uint32_t(value >> 32);
      }
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::ldexp_f64_store_x86_64_v4(exec_mask, d, result);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  rocjitsu::arith::ldexp_f64_store_x86_64_v3(exec_mask, d, result);
#endif
  return RJ_ARITH_SUCCESS;
}
#endif

// Scale FP32/FP64 with guest controls, preserving inactive destinations and aliases.
// Reuse integer scaling for scalar/AVX2 FP32; native AVX-512 remains the fast Loose route.
template <bool Fp64, bool Rdna3 = false, bool Cdna = false>
int ldexp(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~known,
          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_MASK |
              (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    instruction_flags = uint32_t(rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags));
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  [[maybe_unused]] const bool loose =
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_LOOSE &&
      !(flags & RJ_ARITH_FP_ROUND_MASK);
  const bool quiet_snan =
      Cdna || !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE) || !rocjitsu::arith::is_exact(flags);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if (loose) {
      if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                   RJ_ARITH_FP_DX10_CLAMP_OFF)) {
        rocjitsu::arith::ldexp_guest_x86_64_v4(Fp64, flags, exec_mask, instruction_flags, d, a,
                                               b[0]);
      } else {
        rocjitsu::arith::ldexp_x86_64_v4(Fp64, exec_mask, instruction_flags, d, a, b[0]);
      }
    } else if constexpr (Fp64) {
      return ldexp_f64_simd(flags, exec_mask, instruction_flags, d, a, b, quiet_snan);
    } else {
      rocjitsu::arith::ldexp_f32_exact_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0],
                                                 b[0], quiet_snan);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Fp64) {
      if (!loose) {
        return ldexp_f64_simd(flags, exec_mask, instruction_flags, d, a, b, quiet_snan);
      }
      // Keep unusual Loose FP64 scales in the general SIMD kernel instead of lane repairs.
      rocjitsu::arith::ldexp_f64_loose_x86_64_v3(flags, exec_mask, instruction_flags, d, a, b[0]);
    } else {
      rocjitsu::arith::ldexp_f32_exact_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0],
                                                 b[0], quiet_snan);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  return ldexp_rounded<Fp64>(flags, exec_mask, instruction_flags, d, a, b, quiet_snan);
}

// Scale FP32/FP64 by powers of two for Wave64. Reject invalid controls before changing outputs.
template <bool Fp64, bool Rdna3 = false, bool Cdna = false>
int ldexp_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  constexpr auto instruction =
      Cdna   ? (Fp64 ? rj_arith_v_ldexp_f64_w32_gfx942 : rj_arith_v_ldexp_f32_w32_gfx942)
      : Fp64 ? (Rdna3 ? rj_arith_v_ldexp_f64_w32 : rj_arith_v_ldexp_f64_w32_gfx12)
             : (Rdna3 ? rj_arith_v_ldexp_f32_w32 : rj_arith_v_ldexp_f32_w32_gfx12);

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          instruction(flags, 0, instruction_flags, nullptr, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Fp64) {
    // Whole-instruction validation is complete; the private FP64 dispatcher needs resolved flags.
    flags = rocjitsu::arith::resolve_compatibility(
        flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest[Fp64 ? 2 : 1];
    const uint32_t *source[Fp64 ? 2 : 1];
    for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
      dest[reg] = d[reg] + half * 32;
      source[reg] = a[reg] + half * 32;
    }
    // The signed integer exponent occupies one VGPR even for FP64.
    const uint32_t *exponent = b[0] + half * 32;
    int error;
    if constexpr (Fp64) {
      error = ldexp<true, Rdna3, Cdna>(flags, uint32_t(exec_mask >> (half * 32)),
                                       uint32_t(half_instruction_flags), dest, source, &exponent);
    } else {
      // FP32 keeps the public DPP operand adapter; FP64 rejects DPP above.
      error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags, dest,
                          source, &exponent, excp_flag_user);
    }
    if (error) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// The inherited scalar model is shared without compiling its definition for SIMD ISAs.
[[gnu::noinline]] uint32_t
rocjitsu::arith::ldexp_f32_value_baseline(uint32_t input, int32_t adjustment, uint64_t flags,
                                          uint32_t instruction_flags, bool quiet_snan) {
  return ldexp_value<32>(input, adjustment, flags, instruction_flags, quiet_snan);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return ldexp<false>(flags, exec_mask, uint32_t(instruction_flags), d, source, b);
        });
  }
  return ldexp<false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ldexp_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return ldexp_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return ldexp<false, true>(flags, exec_mask, uint32_t(instruction_flags), d, source, b);
        });
  }
  return ldexp<false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  return ldexp_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, trapsts);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  return ldexp_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, trapsts);
}

// CDNA retains inherited signaling-NaN quieting; gfx942 also serves gfx950.
int rj_arith_v_ldexp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<false, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// CDNA retains inherited signaling-NaN quieting; gfx942 also serves gfx950.
int rj_arith_v_ldexp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const uint64_t requested_flags = flags;
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp_wave64<false, true, true>(requested_flags, exec_mask, instruction_flags, d, a, b,
                                         trapsts);
}

// CDNA retains inherited signaling-NaN quieting; gfx942 also serves gfx950.
int rj_arith_v_ldexp_f64_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp<true, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// CDNA retains inherited signaling-NaN quieting; gfx942 also serves gfx950.
int rj_arith_v_ldexp_f64_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const uint64_t requested_flags = flags;
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return ldexp_wave64<true, true, true>(requested_flags, exec_mask, instruction_flags, d, a, b,
                                        trapsts);
}
