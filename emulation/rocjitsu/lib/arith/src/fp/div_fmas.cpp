// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Division FMA with conditional scaling and guest rounding/status handling.
// Owns public entry points and their validation/dispatch policy.

#include "common/internal.h"
#include "common/simd.h"
#include "fp/div_fmas_impl.h"
#include "fp/float_output.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate all source lanes before masked stores, including overlapping VGPRs.
template <int Width>
[[gnu::noinline]] void scalar(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t condition, uint32_t *exceptions,
                              bool flush_output, unsigned rounding) {
  using T = typename rocjitsu::arith::IeeeFormat<Width>::Bits;
  T result[32];
  for (int lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    result[lane] = rocjitsu::arith::division_fmas<Width>(
        av, bv, cv, (condition >> lane) & 1, mode, ((exec_mask >> lane) & 1) ? exceptions : nullptr,
        rounding);
  }
  if (flush_output) {
    for (auto &value : result) {
      if (!(value & rocjitsu::arith::IeeeFormat<Width>::infinity)) {
        value &= rocjitsu::arith::IeeeFormat<Width>::sign;
      }
    }
  }
  for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
      }
    }
  }
}

// Evaluate division FMA with conditional scaling for the selected lanes. Reject invalid
// controls before changing outputs.
template <int Width, bool Rdna3>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t condition, uint32_t *fp_status) {
  // RDNA3 output scaling is disabled with IEEE or preserved output denormals.
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~RJ_ARITH_ALU_OMOD_MASK;
    }
  }
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                RJ_ARITH_FP_ROUND_MASK;
  if (rocjitsu::arith::reports_fp_status(flags) &&
      (Rdna3 || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_ROUND_MASK) ||
       !rocjitsu::arith::is_exact(flags))) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // The upstream integer model supplies exact unscaled results on both GPUs.
  // RDNA3 active OMOD has no borrowed exact model; retain the loose path.
  constexpr uint32_t instruction_controls =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_A |
      RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, mode & ~instruction_controls, fp_flags)) {
    return error;
  }
  if (Rdna3 && (mode & RJ_ARITH_ALU_OMOD_MASK) &&
      (flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  bool flush_output = flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (Width == 32) {
    if (rocjitsu::arith::is_exact(flags) && !condition &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4) &&
        rocjitsu::arith::div_fmas_exact_x86_64_v4(flags, exec_mask, d[0], a[0], b[0], c[0],
                                                  fp_status, mode)) {
      return RJ_ARITH_SUCCESS;
    }
  }
  if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::div_fmas_x86_64_v4<Width>(exec_mask, mode, d, a, b, c, condition, flush_output,
                                               report ? fp_status : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::div_fmas_x86_64_v3<Width>(exec_mask, mode, d, a, b, c, condition, flush_output,
                                               report ? fp_status : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
  scalar<Width>(exec_mask, mode, d, a, b, c, condition, report ? &exceptions : nullptr,
                flush_output, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
  if (report) {
    *fp_status |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate division FMA with conditional scaling for Wave64. Reject invalid controls before
// changing outputs.
template <int Width, bool Rdna3>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint64_t condition, uint32_t *fp_status) {
  if (int error =
          run<Width, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, 0, fp_status)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);

  // Consume both halves before updating a potentially aliased global register.
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest[Width / 32];
    const uint32_t *av[Width / 32], *bv[Width / 32], *cv[Width / 32];
    for (int reg = 0; reg < Width / 32; ++reg) {
      dest[reg] = d[reg] + half * 32;
      av[reg] = a[reg] + half * 32;
      bv[reg] = b[reg] + half * 32;
      cv[reg] = c[reg] + half * 32;
    }
    if (int error =
            run<Width, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest, av, bv, cv,
                              uint32_t(condition >> (half * 32)), report ? &exceptions : nullptr)) {
      return error;
    }
  }
  if (report) {
    *fp_status |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Execute DIV_FMAS_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t condition, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_MASK) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

// Execute DIV_FMAS_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t condition, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_MASK) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

// Execute DIV_FMAS_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint64_t condition, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_MASK) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

// Execute DIV_FMAS_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint64_t condition, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_MASK) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

// Execute DIV_FMAS_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t condition,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<32, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition, excp_flag_user);
}

// Execute DIV_FMAS_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint32_t condition,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<64, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition, excp_flag_user);
}

// Execute DIV_FMAS_F32 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint64_t condition,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<32, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition,
                               excp_flag_user);
}

// Execute DIV_FMAS_F64 for the selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_div_fmas_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c, uint64_t condition,
                                      uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<64, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition,
                               excp_flag_user);
}
