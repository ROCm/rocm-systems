// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division-result fixup, special values, and optional guest exception causes.
// Owns public entry points and their validation/dispatch policy.

#include "common/validation.h"
#include "fp/div_fixup_impl.h"
#include "fp/float_output.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <stdint.h>

namespace {

// Fix up division results and special values for the selected lanes. Requires validated
// controls; preserve inactive destination lanes.
template <int Width, bool Rdna3 = false, bool Cdna = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user);

// Flush only denominator/numerator subnormals, retaining their signs and all
// provisional-quotient bits. All source reads precede the instruction's stores.
// rocjitsu's classify_div_fixup applies this same flush before classifying
// exceptional arithmetic, with input-denormal causes suppressed. Classifying
// these already-flushed operands implements that rule without a second model.
template <int Width, bool Rdna3, bool Cdna>
[[gnu::noinline]] int run_flushed(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using T = typename F::Bits;
  uint32_t words[2][(Width == 64 ? 2 : 1)][32];
  const uint32_t *sources[2][(Width == 64 ? 2 : 1)];
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t *const *input = operand ? c : b;
    for (int lane = 0; lane < 32; ++lane) {
      T value = input[0][lane];
      if constexpr (Width == 64) {
        value |= T(input[1][lane]) << 32;
      }
      unsigned shift = 0;
      if constexpr (Width == 16) {
        shift = mode & (RJ_ARITH_ALU_HIGH_B << operand) ? 16 : 0;
        value = (value >> shift) & 65535;
      }
      if (!(value & F::infinity)) {
        value &= F::sign;
      }
      if constexpr (Width == 16) {
        value <<= shift;
      }
      for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
        words[operand][reg][lane] = uint32_t(value >> (reg * 32));
      }
    }
    for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
      sources[operand][reg] = words[operand][reg];
    }
  }
  return run<Width, Rdna3, Cdna>(flags & ~RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, exec_mask, mode, d, a,
                                 sources[0], sources[1], excp_flag_user);
}

// Validate controls independently of the requested numerical/reporting policy.
template <int Width, bool Rdna3> int validate_controls(uint64_t flags, uint32_t mode) {
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                RJ_ARITH_FP_ROUND_MASK | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  constexpr uint32_t instruction_controls =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_A |
      RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
      (Width == 16
           ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D
           : 0);
  return rocjitsu::arith::validate(flags, mode & ~instruction_controls, fp_flags);
}

// Fix up division results and special values for the selected lanes. Reject invalid controls
// before changing outputs.
template <int Width, bool Rdna3, bool Cdna>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  if (int error = validate_controls<Width, Rdna3>(flags, mode)) {
    return error;
  }
  if (Width == 16 && (flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::reports_fp_status(flags) && rocjitsu::arith::is_exact(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool cdna_vector = false;
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  cdna_vector = Cdna && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3);
#elif defined(RJ_ARITH_HAVE_X86_64_V4)
  cdna_vector = Cdna && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4);
#endif
  cdna_vector &= Width != 16 || !(flags & RJ_ARITH_FP_ROUND_MASK);
  if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && !cdna_vector) {
    return run_flushed<Width, Rdna3, Cdna>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  if constexpr (Rdna3 && Width != 16) {
    if (report && !cdna_vector) {
      for (int lane = 0; lane < 32; ++lane) {
        if (!((exec_mask >> lane) & 1)) {
          continue;
        }
        using T = typename rocjitsu::arith::IeeeFormat<Width>::Bits;
        T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
        if constexpr (Width == 64) {
          av |= T(a[1][lane]) << 32;
          bv |= T(b[1][lane]) << 32;
          cv |= T(c[1][lane]) << 32;
        }
        if constexpr (Cdna) {
          // Legacy CDNA classifies modified sources regardless of CLAMP/DX10.
          T values[] = {av, bv, cv};
          using F = rocjitsu::arith::IeeeFormat<Width>;
          for (int i = 0; i < 3; ++i) {
            if (mode & (RJ_ARITH_ALU_ABS_A << i)) {
              values[i] &= F::sign - 1;
            }
            if (mode & (RJ_ARITH_ALU_NEG_A << i)) {
              values[i] ^= F::sign;
            }
          }
          exceptions |= rocjitsu::arith::fixup_legacy_exceptions<Width>(
              values[0], values[1], values[2],
              mode & ~(RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                       RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C),
              flags);
        } else {
          exceptions |= rocjitsu::arith::fixup_rdna3_exceptions<Width>(av, bv, cv, mode, flags);
        }
      }
    }
  }
  bool saturate = flags & RJ_ARITH_FP16_OVFL;
  // CDNA always quiets selected NaNs. In the shared backend IEEE selects only
  // this numeric policy; architecture-specific OMOD gating happens before run.
  // FP16 SIMD final packing uses nearest rounding; directed modes use the scalar packer.
  const uint64_t numeric_flags = flags | (Cdna ? RJ_ARITH_FP_IEEE_MODE : 0);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((Width != 16 || !(flags & RJ_ARITH_FP_ROUND_MASK)) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    // Read status operands before the numeric stores, which may alias sources.
    if (report && (!Rdna3 || Cdna)) {
      exceptions =
          rocjitsu::arith::fixup_exceptions_x86_64_v4<Width>(exec_mask, mode, a, b, c, flags, Cdna);
    }
    rocjitsu::arith::fixup_x86_64_v4<Width, Rdna3>(exec_mask, mode, numeric_flags, d, a, b, c,
                                                   Cdna);
    if (report) {
      *excp_flag_user |= exceptions;
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((Width != 16 || !(flags & RJ_ARITH_FP_ROUND_MASK)) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if (report && (!Rdna3 || Cdna)) {
      exceptions =
          rocjitsu::arith::fixup_exceptions_x86_64_v3<Width>(exec_mask, mode, a, b, c, flags, Cdna);
    }
    rocjitsu::arith::fixup_x86_64_v3<Width, Rdna3>(exec_mask, mode, numeric_flags, d, a, b, c,
                                                   Cdna);
    if (report) {
      *excp_flag_user |= exceptions;
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  using T = typename rocjitsu::arith::IeeeFormat<Width>::Bits;
  T result[32];
  for (int lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    if constexpr (Width == 16) {
      av = (av >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)) & 65535;
      cv = (cv >> (mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0)) & 65535;
    }
    result[lane] = rocjitsu::arith::fixup_value<Width, Rdna3>(
        av, bv, cv, mode, saturate, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24),
        flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, flags & RJ_ARITH_FP_DX10_CLAMP_OFF,
        Cdna || (flags & RJ_ARITH_FP_IEEE_MODE), Cdna);
    if (report && !Rdna3 && ((exec_mask >> lane) & 1)) {
      exceptions |= rocjitsu::arith::fixup_exceptions<Width>(av, bv, cv, mode);
    }
  }
  for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        if constexpr (Width == 16) {
          unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
          d[0][lane] = (d[0][lane] & ~(65535u << shift)) | (result[lane] << shift);
        } else {
          d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
        }
      }
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Fix up division results and special values for Wave64. Reject invalid controls before
// changing outputs.
template <int Width, bool Rdna3 = false, bool Cdna = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  constexpr auto instruction =
      Cdna          ? (Width == 16   ? rj_arith_v_div_fixup_f16_w32_gfx942
                       : Width == 32 ? rj_arith_v_div_fixup_f32_w32_gfx942
                                     : rj_arith_v_div_fixup_f64_w32_gfx942)
      : Width == 16 ? (Rdna3 ? rj_arith_v_div_fixup_f16_w32 : rj_arith_v_div_fixup_f16_w32_gfx12)
      : Width == 32 ? (Rdna3 ? rj_arith_v_div_fixup_f32_w32 : rj_arith_v_div_fixup_f32_w32_gfx12)
                    : (Rdna3 ? rj_arith_v_div_fixup_f64_w32 : rj_arith_v_div_fixup_f64_w32_gfx12);

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = !(Width == 16 && Rdna3) &&
                rocjitsu::arith::reports_fp_status(rocjitsu::arith::resolve_compatibility(
                    flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL));

  // Delay the global-register write until all potentially aliased inputs are read.
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Width == 64 ? 2 : 1];
    const uint32_t *av[Width == 64 ? 2 : 1], *bv[Width == 64 ? 2 : 1], *cv[Width == 64 ? 2 : 1];
    for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
      dest[reg] = d[reg] + half * 32;
      av[reg] = a[reg] + half * 32;
      bv[reg] = b[reg] + half * 32;
      cv[reg] = c[reg] + half * 32;
    }
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                dest, av, bv, cv, report ? &exceptions : nullptr)) {
      return error;
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Resolve inherited CDNA behavior without advertising a new hardware-qualified model.
// Older CDNA gates OMOD; all CDNA quiets NaNs. Only wide Compatibility reports status.
template <int Width, bool Older>
int run_cdna(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
             uint32_t *status) {
  const auto semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  flags = rocjitsu::arith::resolve_compatibility(
      flags, Width == 16 ? RJ_ARITH_SEMANTICS_LOOSE : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate_controls<Width, Older>(flags, uint32_t(mode))) {
    return error;
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (Older && (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE))) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return run<Width, true, true>(flags, exec_mask, uint32_t(mode), d, a, b, c, status);
}

} // namespace

// Execute DIV_FIXUP_F16 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return run<16>(flags, exec_mask,
                                                         uint32_t(instruction_flags), d, source, b,
                                                         c, excp_flag_user);
                                        });
  }
  return run<16>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F16 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<16>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Execute DIV_FIXUP_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts) {
  // The legacy classifier is not qualified for strict reporting with DX10 clear.
  // Validate all controls first, without reading operands or writing outputs.
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
      (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    if (instruction_flags >> 32) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
    if (int error = run<32, true>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr,
                                  nullptr)) {
      return error;
    }
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return run<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Execute DIV_FIXUP_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts) {
  // Preserve Compatibility for the Wave32 entry point's status policy check.
  return run_wave64<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Execute DIV_FIXUP_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts) {
  // The legacy classifier is not qualified for strict reporting with DX10 clear.
  // Validate all controls first, without reading operands or writing outputs.
  if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
      (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    if (instruction_flags >> 32) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
    if (int error = run<64, true>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr,
                                  nullptr)) {
      return error;
    }
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return run<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Execute DIV_FIXUP_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts) {
  // Preserve Compatibility for the Wave32 entry point's status policy check.
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Execute DIV_FIXUP_F16 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  const bool active_omod = (instruction_flags & RJ_ARITH_ALU_OMOD_MASK) &&
                           (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
                           !(flags & RJ_ARITH_FP_IEEE_MODE);
  flags = rocjitsu::arith::resolve_compatibility(
      flags,
      active_omod ? RJ_ARITH_SEMANTICS_LOOSE : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate_controls<16, true>(flags, uint32_t(instruction_flags))) {
    return error;
  }
  if (!rocjitsu::arith::valid_dpp(instruction_flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (active_omod &&
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return run<16, true>(flags, exec_mask,
                                                               uint32_t(instruction_flags), d,
                                                               source, b, c, nullptr);
                                        });
  }
  return run<16, true>(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

// Execute DIV_FIXUP_F16 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fixup_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *trapsts) {
  // Wave32 resolves Compatibility after guest MODE masks OMOD.
  return run_wave64<16, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<16, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_wave64<16, true, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<16, false>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_wave64<32, true, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<32, false>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f64_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f64_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *fp_status) {
  return run_wave64<64, true, true>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// Repair provisional quotients using the inherited CDNA policy.
int rj_arith_v_div_fixup_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c, uint32_t *fp_status) {
  return run_cdna<64, false>(flags, exec_mask, instruction_flags, d, a, b, c, fp_status);
}

// The inherited legacy and ordinary CDNA half FIXUP paths use identical arithmetic.
int rj_arith_v_div_fixup_legacy_f16_w32(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *trapsts) {
  return rj_arith_v_div_fixup_f16_w32_gfx942(flags, exec_mask, instruction_flags, d, a, b, c,
                                             trapsts);
}

// The inherited legacy and ordinary CDNA half FIXUP paths use identical arithmetic.
int rj_arith_v_div_fixup_legacy_f16_w64(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c, uint32_t *trapsts) {
  return rj_arith_v_div_fixup_f16_w64_gfx942(flags, exec_mask, instruction_flags, d, a, b, c,
                                             trapsts);
}
