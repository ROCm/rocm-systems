// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx11/gfx12 FP32 FMA with guest modifiers, rounding, and DPP routing.
// Owns public entry points and their validation/dispatch policy.

#include "common/alu.h"
#include "common/internal.h"
#include "common/simd.h"
#include "fp/fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Exact FMA, including source modifiers, output scaling, CLAMP and DX9 rules.
// Sources may overlap the destination. Requires the default host FP environment.
template <bool Rdna3, bool Dx9Zero>
[[gnu::noinline]] int fma_exact(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Dx9Zero) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::fma_exact_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                         c[0], Dx9Zero, Rdna3);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3) &&
      !(flags & RJ_ARITH_FP_ROUND_MASK)) {
    rocjitsu::arith::fma_exact_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                         c[0], Dx9Zero, Rdna3);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::fma_instruction_value(a[0][lane], b[0][lane], c[0][lane], flags,
                                                          instruction_flags, Rdna3, Dx9Zero);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP32 fused multiply-add for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <bool Rdna3, bool Dx9Zero = false>
int fma(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Rdna3) {
    // DX9 forces output flushing even when the guest requests preserved denormals.
    uint64_t output_flags = flags | (Dx9Zero ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
    instruction_flags =
        uint32_t(rocjitsu::arith::rdna3_output_modifiers(output_flags, instruction_flags));
  }

  if (rocjitsu::arith::is_exact(flags)) {
    return fma_exact<Rdna3, Dx9Zero>(flags, exec_mask, instruction_flags, d, a, b, c);
  }

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if constexpr (Dx9Zero) {
      rocjitsu::arith::fma_dx9_zero_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                              c[0]);
    } else {
      rocjitsu::arith::fma_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Dx9Zero) {
      rocjitsu::arith::fma_dx9_zero_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                              c[0]);
    } else {
      rocjitsu::arith::fma_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    const uint64_t input_flags = flags | (Dx9Zero ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0);
    float x = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(a[0][lane], instruction_flags, input_flags));
    float y = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(b[0][lane], instruction_flags >> 1, input_flags));
    float z = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(c[0][lane], instruction_flags >> 2, input_flags));

    if constexpr (Dx9Zero) {
      if (x == 0 || y == 0) {
        x = 0;
        y = 1;
      }
    }
    float value = __builtin_fmaf(x, y, z);
    if (Dx9Zero || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      value = rocjitsu::arith::flush_denorm_f32(value);
    }
    value = rocjitsu::arith::scale_omod_f32(value, instruction_flags);
    uint32_t bits = rocjitsu::arith::as_bits(value);
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      bits = rocjitsu::arith::clamp_f32_bits(bits, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    result[lane] = bits;
  }

  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Validate and evaluate FP32 FMA, including supported DPP source routing; errors preserve D.
template <bool Rdna3 = false, bool Dx9Zero = false>
int fma_with_dpp(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, uint32_t(instruction_flags) & ~rocjitsu::arith::kFma32InstructionFlags,
          rocjitsu::arith::kFmaGuestFpFlags | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t selected, const uint32_t *const *source) {
                                          return fma_with_dpp<Rdna3, Dx9Zero>(
                                              flags, selected, uint32_t(instruction_flags), d,
                                              source, b, c);
                                        });
  }
  if (!Dx9Zero && rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  return fma<Rdna3, Dx9Zero>(flags, exec_mask, uint32_t(instruction_flags), d, a, b, c);
}

// Keep the scalar scratch space and integer FMA work off the SIMD dispatch path.
template <rocjitsu::arith::FmaOperands Operands, bool Rdna3>
[[gnu::noinline]] int literal_fma_scalar(uint64_t flags, uint32_t exec_mask, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t literal) {
  bool exact = rocjitsu::arith::is_exact(flags);
  uint32_t result[32];
  float k = rocjitsu::arith::as_float(literal);
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    k = rocjitsu::arith::flush_denorm_f32(k);
  }
  for (int lane = 0; lane < 32; ++lane) {
    float x = rocjitsu::arith::as_float(a[0][lane]), y = rocjitsu::arith::as_float(b[0][lane]);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = rocjitsu::arith::flush_denorm_f32(x);
      y = rocjitsu::arith::flush_denorm_f32(y);
    }
    if (exact) {
      result[lane] = (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral)
                         ? rocjitsu::arith::fma_value_f32(rocjitsu::arith::as_bits(x),
                                                          rocjitsu::arith::as_bits(k),
                                                          rocjitsu::arith::as_bits(y), flags, Rdna3)
                         : rocjitsu::arith::fma_value_f32(
                               rocjitsu::arith::as_bits(x), rocjitsu::arith::as_bits(y),
                               rocjitsu::arith::as_bits(k), flags, Rdna3);
      continue;
    }
    float value = (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral)
                      ? __builtin_fmaf(x, k, y)
                      : __builtin_fmaf(x, y, k);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      value = rocjitsu::arith::flush_denorm_f32(value);
    }
    result[lane] = rocjitsu::arith::as_bits(value);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP32 fused multiply-add for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::FmaOperands Operands, bool Rdna3 = false>
int literal_fma(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  static_assert(Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral ||
                Operands == rocjitsu::arith::FmaOperands::AddLiteral);
  if (int error = rocjitsu::arith::validate(flags, instruction_flags,
                                            rocjitsu::arith::kFmaGuestFpFlags |
                                                (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  [[maybe_unused]] bool exact = rocjitsu::arith::is_exact(flags);
#if defined(RJ_ARITH_HAVE_X86_64_V3) || defined(RJ_ARITH_HAVE_X86_64_V4)
  if (exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    uint32_t broadcast[32];
    for (auto &value : broadcast) {
      value = literal;
    }
    const uint32_t *bb = (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral) ? broadcast
                                                                                     : b[0],
                   *cc = (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral) ? b[0]
                                                                                     : broadcast;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::fma_exact_x86_64_v4(flags, exec_mask, 0, d[0], a[0], bb, cc, false, Rdna3);
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (!(flags & RJ_ARITH_FP_ROUND_MASK)) {
      rocjitsu::arith::fma_exact_x86_64_v3(flags, exec_mask, 0, d[0], a[0], bb, cc, false, Rdna3);
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (!exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::literal_fma_x86_64_v4<Operands>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::literal_fma_x86_64_v3<Operands>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  return literal_fma_scalar<Operands, Rdna3>(flags, exec_mask, d, a, b, literal);
}

// Evaluate FP32 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <bool Rdna3, bool Dx9Zero = false>
int fma_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  int error =
      fma_with_dpp<Rdna3, Dx9Zero>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr);
  if (error || !exec_mask) {
    return error;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    uint32_t half_exec_mask = uint32_t(exec_mask >> (half * 32));
    error = fma_with_dpp<Rdna3, Dx9Zero>(flags, half_exec_mask, half_instruction_flags, &dest, &x,
                                         &y, &z);
    if (error) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP32 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::FmaOperands Operands, bool Rdna3 = false>
int literal_fma_wave64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t literal) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = literal_fma<Operands, Rdna3>(flags, 0, instruction_flags, nullptr, nullptr,
                                               nullptr, literal)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error = literal_fma<Operands, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)),
                                                 instruction_flags, &dest, &x, &y, literal)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_with_dpp(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_with_dpp<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  // FMAC has no explicit C operand, so its modifiers are invalid.
  if (instruction_flags & (RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_C)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_with_dpp(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma<rocjitsu::arith::FmaOperands::MultiplyLiteral>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma<rocjitsu::arith::FmaOperands::AddLiteral>(flags, exec_mask, instruction_flags,
                                                               d, a, b, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  // FMAC has no explicit C operand, so its modifiers are invalid.
  if (instruction_flags & (RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_C)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                   const uint32_t *const *b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma_wave64<rocjitsu::arith::FmaOperands::MultiplyLiteral>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma_wave64<rocjitsu::arith::FmaOperands::AddLiteral>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_with_dpp<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_wave64<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  // FMAC has no explicit C operand, so its modifiers are invalid.
  if (instruction_flags & (RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_C)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_with_dpp<true>(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmac_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  // FMAC has no explicit C operand, so its modifiers are invalid.
  if (instruction_flags & (RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_C)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_wave64<true, false>(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_with_dpp<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fmac_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (instruction_flags & ~uint64_t(known)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fma_dx9_zero_f32_w32(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Accumulate fused A*B into D with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fmac_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (instruction_flags & ~uint64_t(known)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fma_dx9_zero_f32_w64(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma<rocjitsu::arith::FmaOperands::MultiplyLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t literal, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma<rocjitsu::arith::FmaOperands::AddLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmamk_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                             const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma_wave64<rocjitsu::arith::FmaOperands::MultiplyLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fmaak_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t literal, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return literal_fma_wave64<rocjitsu::arith::FmaOperands::AddLiteral, true>(
      flags, exec_mask, instruction_flags, d, a, b, literal);
}
