// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 scaling by powers of two and integer exponent extraction.
// Owns public entry points and their validation/dispatch policy.

#include "fp/half_exponent.h"
#include "common/alu.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "common/math.h"
#include "fp/half_fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Adapted from rocjitsu fp_mode::ldexp_f16 and its generated VOP3 output
// pipeline. Bounded finite powers of two are exact in double; integer narrowing
// applies guest rounding without the original ScopedFenv guards.
int ldexp_rounded(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  const unsigned a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const unsigned b_shift = instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  const unsigned d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned omod = (instruction_flags >> 6) & 3;
  const double scales[] = {1, 2, 4, 0.5};
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[0][lane] >> a_shift);
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      bits &= 0x7fff;
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      bits ^= 0x8000;
    }
    const unsigned magnitude = bits & 0x7fff;
    uint16_t encoded;
    if (magnitude >= 0x7c00) {
      encoded = bits | (magnitude > 0x7c00 ? 0x200 : 0);
      if ((instruction_flags & RJ_ARITH_ALU_CLAMP) &&
          !(magnitude > 0x7c00 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
        encoded = rocjitsu::arith::half_fma_clamp(encoded);
      }
    } else {
      if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && magnitude < 0x400) {
        bits &= 0x8000;
      }
      const uint16_t raw_exponent = uint16_t(b[0][lane] >> b_shift);
      const int exponent = int(raw_exponent) - (raw_exponent & 0x8000 ? 65536 : 0);
      double value = rocjitsu::arith::ldexp(double(rocjitsu::arith::f16_to_float(bits)),
                                            rocjitsu::arith::clamp(exponent, -64, 64));
      const bool tiny = __builtin_fabs(value) < 0x1p-14;
      // Round the instruction result before output scaling. Overflow must not
      // be rescued by OMOD_HALF, nor a tiny result by OMOD_2/4.
      encoded = rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(value),
                                                 flags & RJ_ARITH_FP16_OVFL, rounding);
      if (tiny && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        encoded &= 0x8000;
      }
      if (omod) {
        value = tiny ? 0.0 : double(rocjitsu::arith::f16_to_float(encoded)) * scales[omod];
        encoded = rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(value),
                                                   flags & RJ_ARITH_FP16_OVFL, rounding);
        if (__builtin_fabs(value) < 0x1p-14) {
          encoded &= 0x8000;
        }
      }
      if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
        encoded = rocjitsu::arith::half_fma_clamp(encoded);
      }
    }
    result[lane] = encoded;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 LDEXP with guest rounding/denormal controls, preserving inactive destinations.
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, bool rdna3);

// Scale or decompose FP16 exponents for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Ldexp, bool GuestDenormals = false>
int half_exponent(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  bool rdna3 = false) {
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return half_exponent<Ldexp, GuestDenormals>(
                                              flags, exec_mask, uint32_t(instruction_flags), d,
                                              source, b, rdna3);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_HALF |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D |
                         (Ldexp ? RJ_ARITH_ALU_HIGH_B : 0);
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~known,
          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (Ldexp ? RJ_ARITH_FP_ROUND_MASK : 0) |
              (rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (rdna3 && Ldexp) {
    instruction_flags = rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags);
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Ldexp) {
    if (rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) {
      return ldexp_rounded(flags, exec_mask, uint32_t(instruction_flags), d, a, b);
    }
    if constexpr (!GuestDenormals) {
      if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                   RJ_ARITH_FP_DX10_CLAMP_OFF)) {
        return ldexp_guest(flags, exec_mask, instruction_flags, d, a, b, rdna3);
      }
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (GuestDenormals) {
      rocjitsu::arith::half_ldexp_guest_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0],
                                                  b[0]);
    } else {
      rocjitsu::arith::half_exponent_x86_64_v3<Ldexp>(flags, exec_mask, instruction_flags, d[0],
                                                      a[0], Ldexp ? b[0] : nullptr);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  int a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[0][lane] >> a_shift);
    if constexpr (Ldexp) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
          (bits & 0x7fff) < 0x400) {
        bits &= 0x8000;
      }
      uint16_t raw_exponent = uint16_t(b[0][lane] >> b_shift);
      int exponent = int(raw_exponent) - (raw_exponent & 0x8000 ? 65536 : 0);

      // Following rocjitsu's F16 LDEXP model, bound the exponent while retaining
      // every FP16 rounding outcome, including output scaling and saturation.
      float value = rocjitsu::arith::ldexp(
          rocjitsu::arith::alu_input(rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(bits)),
                                     instruction_flags),
          rocjitsu::arith::clamp(exponent, -64, 64));
      const bool tiny_before_rounding = __builtin_fabs(value) < 0x1p-14f;
      if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
        // Test tininess before rounding; finite overflow is rounded before OMOD.
        value = __builtin_fabs(value) < 0x1p-14f
                    ? 0.0f
                    : rocjitsu::arith::f16_to_float(
                          rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL));
      }
      result[lane] = rocjitsu::arith::float_to_f16(
          rocjitsu::arith::alu_output_f16(value, (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
                                                         __builtin_isnan(value)
                                                     ? instruction_flags & ~RJ_ARITH_ALU_CLAMP
                                                     : instruction_flags),
          flags & RJ_ARITH_FP16_OVFL);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && tiny_before_rounding) {
        result[lane] &= 0x8000;
      }
    } else {
      // Sign modifiers do not affect the exponent; OMOD is ignored for integer
      // results and all FP16 exponents fit int16_t, so CLAMP has no effect.
      int field = (bits >> 10) & 31, fraction = bits & 1023;
      int exponent = field - 14;
      if (field == 31 || (bits & 0x7fff) == 0 ||
          (field == 0 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS))) {
        exponent = 0;
      } else if (field == 0) {
        exponent = -24;
        while (fraction) {
          ++exponent;
          fraction >>= 1;
        }
      }
      result[lane] = uint16_t(exponent);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 LDEXP with guest rounding/denormal controls, preserving inactive destinations.
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, bool rdna3) {
  // Separate entry prevents loop merging from adding guest checks to default arithmetic.
  return half_exponent<true, true>(flags, exec_mask, instruction_flags, d, a, b, rdna3);
}

// Scale or decompose FP16 exponents for Wave64. Reject invalid controls before changing
// outputs.
template <bool Ldexp>
int half_exponent_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         bool rdna3 = false) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          half_exponent<Ldexp>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, rdna3)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    const uint32_t *exponent = Ldexp ? b[0] + half * 32 : nullptr;
    if (int error =
            half_exponent<Ldexp>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                 &dest, &source, &exponent, rdna3)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return half_exponent<true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Extract the integer exponent for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a,
                                           [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_exponent<false>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
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
  return half_exponent_wave64<true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Extract the integer exponent for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                           uint64_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a,
                                           [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_exponent_wave64<false>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
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
  return half_exponent<true>(flags, exec_mask, instruction_flags, d, a, b, true);
}

// Scale A by a power of two specified by B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ldexp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
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
  return half_exponent_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, true);
}

// Extract the integer exponent for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // Integer exponent output is independent of the RDNA3 NaN clamp policy.
  return rj_arith_v_frexp_exp_i16_f16_w32_gfx12(flags & ~RJ_ARITH_FP_DX10_CLAMP_OFF, exec_mask,
                                                instruction_flags, d, a, trapsts);
}

// Extract the integer exponent for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_frexp_exp_i16_f16_w64_gfx12(flags & ~RJ_ARITH_FP_DX10_CLAMP_OFF, exec_mask,
                                                instruction_flags, d, a, trapsts);
}
