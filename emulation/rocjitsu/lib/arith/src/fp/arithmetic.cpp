// SPDX-License-Identifier: MIT

// FP32 FMA/FMAC and literal-operand variants, including gfx11 and DX9-zero behavior.
// Owns public entry points and their validation/dispatch policy.
// DX9 flushing and zero-product addition follow rocjitsu fp_mode::arithmetic.

#include "common/alu.h"
#include "common/exact_arithmetic.h"
#include "common/internal.h"
#include "common/simd.h"
#include "fp/fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Exact FMA, including source modifiers, output scaling, CLAMP and DX9 rules.
// Sources may overlap the destination. Requires the default host FP environment.
template <bool Rdna3, bool Dx9Zero>
[[gnu::noinline]] int fma_exact(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Dx9Zero) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4 &&
      rocjitsu::arith::fma_exact_x86_64_v4(flags, exec_mask, mode, d[0], a[0], b[0], c[0], Dx9Zero,
                                           Rdna3)) {
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3 &&
      rocjitsu::arith::fma_exact_x86_64_v3(flags, exec_mask, mode, d[0], a[0], b[0], c[0], Dx9Zero,
                                           Rdna3)) {
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::fma_instruction_value(a[0][lane], b[0][lane], c[0][lane], flags,
                                                          mode, Rdna3, Dx9Zero);
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
template <bool Dx9Zero, bool Rdna3 = false>
int fma(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Rdna3) {
    // DX9 forces output flushing, so guest denormal preservation cannot disable OMOD.
    if ((!Dx9Zero && !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) ||
        (flags & RJ_ARITH_FP_IEEE_MODE)) {
      instruction_flags &= ~RJ_ARITH_ALU_OMOD_HALF;
    }
  }

  if (rocjitsu::arith::is_exact(flags)) {
    return fma_exact<Rdna3, Dx9Zero>(flags, exec_mask, instruction_flags, d, a, b, c);
  }

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
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
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
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
    float x = rocjitsu::arith::alu_input(a[0][lane], instruction_flags);
    float y = rocjitsu::arith::alu_input(b[0][lane], instruction_flags >> 1);
    float z = rocjitsu::arith::alu_input(c[0][lane], instruction_flags >> 2);
    if (Dx9Zero || (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      x = rocjitsu::arith::flush_denorm_f32(x);
      y = rocjitsu::arith::flush_denorm_f32(y);
      z = rocjitsu::arith::flush_denorm_f32(z);
    }
    if constexpr (Dx9Zero) {
      if (x == 0 || y == 0) {
        x = 0;
        y = 1;
      }
    }
    float value = std::fma(x, y, z);
    if (Dx9Zero || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      value = rocjitsu::arith::flush_denorm_f32(value);
    }
    const uint32_t output_mode = (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
                                         (rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000
                                     ? instruction_flags & ~RJ_ARITH_ALU_CLAMP
                                     : instruction_flags;
    result[lane] = rocjitsu::arith::as_bits(rocjitsu::arith::alu_output_f32(value, output_mode));
  }

  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Validates DPP routing and stages A before any destination writes.
template <bool Rdna3, bool Dx9Zero>
[[gnu::noinline]] int fma_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  if (!rocjitsu::arith::valid_dpp(mode)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  uint32_t permuted[32];
  const uint32_t *source = permuted;
  if (mode & RJ_ARITH_DPP8) {
    rocjitsu::arith::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
  } else {
    exec_mask = rocjitsu::arith::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  return fma<Dx9Zero, Rdna3>(flags, exec_mask, uint32_t(mode), d, &source, b, c);
}

// Validate and evaluate FP32 FMA, including supported DPP source routing; errors preserve D.
template <bool Rdna3 = false, bool Dx9Zero = false>
int fma_with_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                 uint32_t known) {
  if (uint32_t(mode) & ~known) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                RJ_ARITH_FP_ROUND_MASK | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, 0, true, fp_flags)) {
    return error;
  }
  if (mode >> 32) {
    return fma_dpp<Rdna3, Dx9Zero>(flags, exec_mask, mode, d, a, b, c);
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  return fma<Dx9Zero, Rdna3>(flags, exec_mask, uint32_t(mode), d, a, b, c);
}

// Keep the scalar scratch space and integer FMA work off the SIMD dispatch path.
template <bool Multiply, bool Rdna3>
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
      result[lane] =
          Multiply
              ? rocjitsu::arith::fma_value_f32(rocjitsu::arith::as_bits(x),
                                               rocjitsu::arith::as_bits(k),
                                               rocjitsu::arith::as_bits(y), false, flags, Rdna3)
              : rocjitsu::arith::fma_value_f32(rocjitsu::arith::as_bits(x),
                                               rocjitsu::arith::as_bits(y),
                                               rocjitsu::arith::as_bits(k), false, flags, Rdna3);
      continue;
    }
    float value = Multiply ? std::fma(x, k, y) : std::fma(x, y, k);
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
template <bool Multiply, bool Rdna3 = false>
int literal_fma(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (int error = rocjitsu::arith::validate(flags, mode, true,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                RJ_ARITH_FP_IEEE_MODE)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  [[maybe_unused]] bool exact = rocjitsu::arith::is_exact(flags);
#if defined(RJ_ARITH_HAVE_X86_64_V3) || defined(RJ_ARITH_HAVE_X86_64_V4)
  if (exact && (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    uint32_t broadcast[32];
    for (auto &value : broadcast) {
      value = literal;
    }
    const uint32_t *bb = Multiply ? broadcast : b[0], *cc = Multiply ? b[0] : broadcast;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4 &&
        rocjitsu::arith::fma_exact_x86_64_v4(flags, exec_mask, 0, d[0], a[0], bb, cc, false,
                                             Rdna3)) {
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (rocjitsu::arith::fma_exact_x86_64_v3(flags, exec_mask, 0, d[0], a[0], bb, cc, false,
                                             Rdna3)) {
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (!exact && (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::literal_fma_x86_64_v4<Multiply>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!exact && (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::literal_fma_x86_64_v3<Multiply>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  return literal_fma_scalar<Multiply, Rdna3>(flags, exec_mask, d, a, b, literal);
}

// Evaluate FP32 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <bool Dx9Zero, bool Rdna3 = false>
int fma_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  int error =
      fma_with_dpp<Rdna3, Dx9Zero>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, 0x1ff);
  if (error || !exec_mask) {
    return error;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    uint32_t half_exec_mask = uint32_t(exec_mask >> (half * 32));
    error =
        fma_with_dpp<Rdna3, Dx9Zero>(flags, half_exec_mask, half_mode, &dest, &x, &y, &z, 0x1ff);
    if (error) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP32 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <bool Multiply, bool Rdna3 = false>
int literal_fma_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          literal_fma<Multiply, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, literal)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error = literal_fma<Multiply, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode,
                                                 &dest, &x, &y, literal)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return fma_with_dpp(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return fma_with_dpp<false, true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  return fma_with_dpp(flags, exec_mask, mode, d, a, b, d, known);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
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
  return literal_fma<true>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
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
  return literal_fma<false>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
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
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return fma_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (uint32_t(mode) & ~known) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_wave64<false>(flags, exec_mask, mode, d, a, b, d);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
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
  return literal_fma_wave64<true>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t literal,
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
  return literal_fma_wave64<false>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return fma_with_dpp<true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return fma_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  return fma_with_dpp<true>(flags, exec_mask, mode, d, a, b, d, known);
}

// Accumulate fused A*B into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmac_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (uint32_t(mode) & ~known) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fma_wave64<false, true>(flags, exec_mask, mode, d, a, b, d);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return fma_with_dpp<true, true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

// Compute fused A*B+C with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fma_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return fma_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate fused A*B into D with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_fmac_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
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
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (instruction_flags & ~uint64_t(known)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_fma_dx9_zero_f32_w64(flags, exec_mask, instruction_flags, d, a, b, d);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, uint32_t literal, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
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
  return literal_fma<true, true>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b, uint32_t literal,
                             [[maybe_unused]] uint32_t *trapsts) {
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
  return literal_fma<false, true>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*literal+B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmamk_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, uint32_t literal, const uint32_t *const *b,
                             [[maybe_unused]] uint32_t *trapsts) {
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
  return literal_fma_wave64<true, true>(flags, exec_mask, mode, d, a, b, literal);
}

// Compute fused A*B+literal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fmaak_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b, uint32_t literal,
                             [[maybe_unused]] uint32_t *trapsts) {
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
  return literal_fma_wave64<false, true>(flags, exec_mask, mode, d, a, b, literal);
}
