// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx11/gfx12 FP16 FMA with guest rounding and optional FP status.
// Owns public entry points and their validation/dispatch policy.

#include "common/alu.h"
#include "common/internal.h"
#include "common/simd.h"
#include "fp/fma.h"
#include "fp/half_fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate FP16 fused multiply-add for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::FmaOperands Operands = rocjitsu::arith::FmaOperands::Registers,
          bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user, uint16_t literal = 0) {
  bool report = rocjitsu::arith::reports_fp_status(flags);
  if (instruction_flags >> 32) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                          [&](uint32_t exec_mask, const uint32_t *const *source) {
                                            return run<Operands, Rdna3>(
                                                flags, exec_mask, uint32_t(instruction_flags), d,
                                                source, b, c, excp_flag_user, literal);
                                          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = Operands == rocjitsu::arith::FmaOperands::Registers
                             ? rocjitsu::arith::kFma16InstructionFlags
                             : RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags |
                                                (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    instruction_flags = rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags);
  }
  if (report && rocjitsu::arith::half_fma_reporting_unsupported(flags, Rdna3)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      rocjitsu::arith::half_fma_exact_x86_64_v4(flags, exec_mask, uint32_t(instruction_flags), d[0],
                                                a[0], b[0], c[0], excp_flag_user, Rdna3);
      return RJ_ARITH_SUCCESS;
    } else {
      // Present the literal as an ordinary SIMD operand; FMAMK moves the original B selector to
      // the addend C slot.
      uint32_t broadcast[32];
      for (auto &value : broadcast) {
        value = literal;
      }
      constexpr bool multiply = Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral;
      uint32_t modifiers = uint32_t(instruction_flags);
      if constexpr (multiply) {
        modifiers = (modifiers & ~RJ_ARITH_ALU_HIGH_B) |
                    ((modifiers & RJ_ARITH_ALU_HIGH_B) ? RJ_ARITH_ALU_HIGH_C : 0);
      }
      rocjitsu::arith::half_fma_exact_x86_64_v4(flags, exec_mask, modifiers, d[0], a[0],
                                                multiply ? broadcast : b[0],
                                                multiply ? b[0] : broadcast, excp_flag_user, Rdna3);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::is_exact(flags) && !(flags & RJ_ARITH_FP_ROUND_MASK) &&
      (!report || (instruction_flags & RJ_ARITH_ALU_CLAMP)) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Operands == rocjitsu::arith::FmaOperands::Registers) {
      rocjitsu::arith::half_fma_exact_x86_64_v3(flags, exec_mask, uint32_t(instruction_flags), d[0],
                                                a[0], b[0], c[0], Rdna3);
      return RJ_ARITH_SUCCESS;
    } else {
      uint32_t broadcast[32];
      for (auto &value : broadcast) {
        value = literal;
      }
      constexpr bool multiply = Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral;
      uint32_t modifiers = uint32_t(instruction_flags);
      if constexpr (multiply) {
        modifiers = (modifiers & ~RJ_ARITH_ALU_HIGH_B) |
                    ((modifiers & RJ_ARITH_ALU_HIGH_B) ? RJ_ARITH_ALU_HIGH_C : 0);
      }
      rocjitsu::arith::half_fma_exact_x86_64_v3(flags, exec_mask, modifiers, d[0], a[0],
                                                multiply ? broadcast : b[0],
                                                multiply ? b[0] : broadcast, Rdna3);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  bool exact = rocjitsu::arith::is_exact(flags);
  if (!exact && !(flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_fma_x86_64_v3<Operands>(
        flags, exec_mask, instruction_flags, d[0], a[0], b[0],
        Operands == rocjitsu::arith::FmaOperands::Registers ? c[0] : nullptr, literal);
    return RJ_ARITH_SUCCESS;
  }
#endif
  int a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = instruction_flags & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
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
        x, y, z, instruction_flags, flags,
        report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr);
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
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user, uint16_t literal = 0) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Operands, Rdna3>(flags, 0, instruction_flags, nullptr, nullptr, nullptr,
                                       nullptr, excp_flag_user, literal)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    const uint32_t *z =
        Operands == rocjitsu::arith::FmaOperands::Registers ? c[0] + half * 32 : nullptr;
    if (int error =
            run<Operands, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                 &dest, &x, &y, &z, report ? &exceptions : nullptr, literal)) {
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
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return rj_arith_v_fmac_f16_w32_gfx12(
                                              flags, exec_mask, uint32_t(instruction_flags), d,
                                              source, b, excp_flag_user);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
                         RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags)) {
    return error;
  }

  // FMAC reads the same destination half that it overwrites.
  if (instruction_flags & RJ_ARITH_ALU_HIGH_D) {
    instruction_flags |= RJ_ARITH_ALU_HIGH_C;
  }
  return run(flags, exec_mask, instruction_flags, d, a, b, d, excp_flag_user);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::FmaOperands::MultiplyLiteral>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr, excp_flag_user, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::FmaOperands::AddLiteral>(flags, exec_mask, instruction_flags, d, a, b,
                                                       nullptr, excp_flag_user, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run_wave64(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if ((instruction_flags >> 32) && !rocjitsu::arith::valid_dpp(instruction_flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
                         RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags) & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags)) {
    return error;
  }

  // FMAC reads the same destination half that it overwrites.
  if (instruction_flags & RJ_ARITH_ALU_HIGH_D) {
    instruction_flags |= RJ_ARITH_ALU_HIGH_C;
  }
  return run_wave64(flags, exec_mask, instruction_flags, d, a, b, d, excp_flag_user);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::MultiplyLiteral>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr, excp_flag_user, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint16_t literal,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::AddLiteral>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr, excp_flag_user, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run<rocjitsu::arith::FmaOperands::Registers, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b, c, trapsts);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run_wave64<rocjitsu::arith::FmaOperands::Registers, true>(
      flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return rj_arith_v_fmac_f16_w32(
                                              flags, exec_mask, uint32_t(instruction_flags), d,
                                              source, b, trapsts);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
                         RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags |
                                                RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }

  // FMAC reads the same destination half that it overwrites.
  if (instruction_flags & RJ_ARITH_ALU_HIGH_D) {
    instruction_flags |= RJ_ARITH_ALU_HIGH_C;
  }
  return run<rocjitsu::arith::FmaOperands::Registers, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b, d, trapsts);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if ((instruction_flags >> 32) && !rocjitsu::arith::valid_dpp(instruction_flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
                         RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags) & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags |
                                                RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }

  // FMAC reads the same destination half that it overwrites.
  if (instruction_flags & RJ_ARITH_ALU_HIGH_D) {
    instruction_flags |= RJ_ARITH_ALU_HIGH_C;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::Registers, true>(
      flags, exec_mask, instruction_flags, d, a, b, d, trapsts);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                             const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::FmaOperands::MultiplyLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts, literal);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                             const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::MultiplyLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<rocjitsu::arith::FmaOperands::AddLiteral, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b, nullptr, trapsts, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint16_t literal, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<rocjitsu::arith::FmaOperands::AddLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts, literal);
}
