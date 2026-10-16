// SPDX-License-Identifier: MIT

// FP16 FMA/FMAC with guest rounding and optional FP status.
// Owns public entry points and their validation/dispatch policy.

#include "fp/half_fma.h"
#include "common/exact_arithmetic.h"
#include "common/internal.h"
#include "fp/fma.h"
#include "fp/half_fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

static const uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                 RJ_ARITH_FP_ROUND_MASK;

// Evaluate FP16 fused multiply-add for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::FmaOperands Operands = rocjitsu::arith::FmaOperands::Registers,
          bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user, uint16_t literal = 0) {
  bool report = rocjitsu::arith::reports_fp_status(flags);
  if (mode >> 32) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return run<Operands, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, b, c,
                                        excp_flag_user, literal);
          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = Operands == rocjitsu::arith::FmaOperands::Registers
                             ? 0x1fffU
                             : RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true,
                                            fp_flags | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
    }
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      if (rocjitsu::arith::half_fma_exact_x86_64_v4(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                                    b[0], c[0], excp_flag_user, Rdna3)) {
        return RJ_ARITH_SUCCESS;
      }
    } else {
      // Present the literal as an ordinary SIMD operand; FMAMK moves the original B selector to
      // the addend C slot.
      uint32_t broadcast[32];
      for (auto &value : broadcast) {
        value = literal;
      }
      constexpr bool multiply = Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral;
      uint32_t modifiers = uint32_t(mode);
      if constexpr (multiply) {
        modifiers = (modifiers & ~RJ_ARITH_ALU_HIGH_B) |
                    ((modifiers & RJ_ARITH_ALU_HIGH_B) ? RJ_ARITH_ALU_HIGH_C : 0);
      }
      if (rocjitsu::arith::half_fma_exact_x86_64_v4(
              flags, exec_mask, modifiers, d[0], a[0], multiply ? broadcast : b[0],
              multiply ? b[0] : broadcast, excp_flag_user, Rdna3)) {
        return RJ_ARITH_SUCCESS;
      }
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      if (rocjitsu::arith::half_fma_exact_x86_64_v3(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                                    b[0], c[0], excp_flag_user, Rdna3)) {
        return RJ_ARITH_SUCCESS;
      }
    } else {
      uint32_t broadcast[32];
      for (auto &value : broadcast) {
        value = literal;
      }
      constexpr bool multiply = Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral;
      uint32_t modifiers = uint32_t(mode);
      if constexpr (multiply) {
        modifiers = (modifiers & ~RJ_ARITH_ALU_HIGH_B) |
                    ((modifiers & RJ_ARITH_ALU_HIGH_B) ? RJ_ARITH_ALU_HIGH_C : 0);
      }
      if (rocjitsu::arith::half_fma_exact_x86_64_v3(
              flags, exec_mask, modifiers, d[0], a[0], multiply ? broadcast : b[0],
              multiply ? b[0] : broadcast, excp_flag_user, Rdna3)) {
        return RJ_ARITH_SUCCESS;
      }
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  bool exact = rocjitsu::arith::is_exact(flags);
  if (!exact && !(flags & RJ_ARITH_FP_ROUND_MASK) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::half_fma_x86_64_v3<Operands>(
        flags, exec_mask, mode, d[0], a[0], b[0],
        Operands == rocjitsu::arith::FmaOperands::Registers ? c[0] : nullptr, literal);
    return RJ_ARITH_SUCCESS;
  }
#endif
  int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t x = uint16_t(a[0][lane] >> a_shift), y = uint16_t(b[0][lane] >> b_shift), z;
    if constexpr (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral) {
      z = y;
      y = literal;
    } else if constexpr (Operands == rocjitsu::arith::FmaOperands::AddLiteral) {
      z = literal;
    } else {
      z = uint16_t(c[0][lane] >> c_shift);
    }
    result[lane] = rocjitsu::arith::half_fma_value<Rdna3>(
        x, y, z, mode, flags, report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::FmaOperands Operands = rocjitsu::arith::FmaOperands::Registers,
          bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user, uint16_t literal = 0) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Operands, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr,
                                       excp_flag_user, literal)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    const uint32_t *z =
        Operands == rocjitsu::arith::FmaOperands::Registers ? c[0] + half * 32 : nullptr;
    if (int error =
            run<Operands, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &x,
                                 &y, &z, report ? &exceptions : nullptr, literal)) {
      return error;
    }
  }
  // The global output may alias an upper-half source: update only after all reads.
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, false)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, false)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run_wave64(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, true)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run<rocjitsu::arith::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b, c,
                                                            trapsts);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, true)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b,
                                                                   c, trapsts);
}
