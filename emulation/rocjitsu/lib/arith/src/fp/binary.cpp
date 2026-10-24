// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 add/subtract/multiply and DX9-zero multiplication.
// Owns public entry points and their validation/dispatch policy.

#include "fp/binary.h"
#include "common/alu.h"
#include "common/internal.h"
#include "common/simd.h"
#include "fp/fma_scalar.h"
#include "fp/mul_exceptions.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Reporting is qualified only for gfx1201 default arithmetic, without OMOD/CLAMP.
template <bool Rdna3> bool mul_status_supported(uint64_t flags, uint64_t mode) {
  return !Rdna3 &&
         !(flags & (RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                    RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) &&
         !(mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP));
}

// rocjitsu fp_mode::arithmetic defines ADD as FMA(A, 1, B) and MUL as
// FMA(A, B, a zero with the product sign). Reuse rj-arith's existing integer FMA
// model, including modifiers and tiny-result scaling, without host FP changes.
template <bool Multiply, bool Rdna3>
[[gnu::noinline]] int binary_exact(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *state) {
  uint32_t extra[32];
  const uint32_t *source = extra;
  auto fma = Rdna3 ? rj_arith_v_fma_f32_w32 : rj_arith_v_fma_f32_w32_gfx12;
  if constexpr (Multiply) {
    for (int lane = 0; lane < 32; ++lane) {
      uint32_t x = mode & RJ_ARITH_ALU_ABS_A ? 0 : a[0][lane];
      uint32_t y = mode & RJ_ARITH_ALU_ABS_B ? 0 : b[0][lane];
      uint32_t negate = ((mode & RJ_ARITH_ALU_NEG_A) ? 0x80000000U : 0) ^
                        ((mode & RJ_ARITH_ALU_NEG_B) ? 0x80000000U : 0);
      extra[lane] = (x ^ y ^ negate) & 0x80000000U;
    }
    uint32_t causes = 0;
    if (rocjitsu::arith::reports_fp_status(flags)) {
      for (int lane = 0; lane < 32; ++lane) {
        if ((exec_mask >> lane) & 1) {
          causes |= rocjitsu::arith::mul_exceptions_f32(a[0][lane], b[0][lane]);
        }
      }
    }
    const int error =
        fma(rocjitsu::arith::without_fp_status(flags), exec_mask, mode, d, a, b, &source, nullptr);
    if (!error && rocjitsu::arith::reports_fp_status(flags)) {
      *state |= causes;
    }
    return error;
  } else {
    for (auto &value : extra) {
      value = 0x3f800000U;
    }
    const uint32_t b_modifiers = RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B;
    mode = (mode & ~b_modifiers) | ((mode & b_modifiers) << 1);
    return fma(rocjitsu::arith::without_fp_status(flags), exec_mask, mode, d, a, &source, b,
               nullptr);
  }
}

// rocjitsu MUL_LEGACY applies guest input flushing, then returns +0 if either
// operand is zero. Reuse the inherited integer multiply for the remaining
// arithmetic; integer source preparation avoids host DAZ and invalid exceptions.
template <bool Rdna3>
int multiply_dx9_exact(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b) {
  uint32_t input[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    for (int operand = 0; operand < 2; ++operand) {
      uint32_t value = (operand ? b : a)[0][lane];
      if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
        value &= 0x7fffffffU;
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
        value ^= 0x80000000U;
      }
      if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && !(value & 0x7f800000U)) {
        value &= 0x80000000U;
      }
      input[operand][lane] = value;
    }
    if (!(input[0][lane] & 0x7fffffffU) || !(input[1][lane] & 0x7fffffffU)) {
      input[0][lane] = 0;
      input[1][lane] = 0x3f800000U;
    }
  }
  const uint32_t *x = input[0], *y = input[1];
  return binary_exact<true, Rdna3>(
      flags, exec_mask, mode & (RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP), d, &x, &y, nullptr);
}

// Subtraction negates the selected source without reordering operands: NaN
// priority follows A then B, including SUBREV. Reuse ADD's guest arithmetic.
template <bool Reverse, bool Rdna3>
int subtract_exact(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  mode ^= Reverse ? RJ_ARITH_ALU_NEG_A : RJ_ARITH_ALU_NEG_B;
  return binary_exact<false, Rdna3>(flags, exec_mask, mode, d, a, b, nullptr);
}

// Evaluate FP32 binary arithmetic for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Binary Op, bool Rdna3 = false>
int binary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, uint32_t *state = nullptr) {
  if (rocjitsu::arith::reports_fp_status(flags) && !mul_status_supported<Rdna3>(flags, mode)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return binary<Op, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, b, state);
        });
  }

  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A |
                         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                RJ_ARITH_FP_ROUND_MASK | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true, fp_flags)) {
    return error;
  }

  // RDNA3 ignores output scaling when denormal outputs are preserved or IEEE
  // mode is enabled. RDNA4 applies it independently of either guest setting.
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
    }
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Sub ||
                Op == rocjitsu::arith::Binary::Subrev) {
    if (rocjitsu::arith::is_exact(flags) && (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3 &&
        rocjitsu::arith::binary_exact_x86_64_v3(Op, flags, exec_mask, uint32_t(mode), d[0], a[0],
                                                b[0])) {
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (Op == rocjitsu::arith::Binary::Mul) {
    if (rocjitsu::arith::is_exact(flags) && (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4 &&
        rocjitsu::arith::mul_exact_x86_64_v4(flags, exec_mask, uint32_t(mode), d[0], a[0], b[0],
                                             state)) {
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  if constexpr (Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Mul) {
    if (rocjitsu::arith::is_exact(flags)) {
      return binary_exact<Op == rocjitsu::arith::Binary::Mul, Rdna3>(
          flags, exec_mask, uint32_t(mode), d, a, b, state);
    }
  }
  if constexpr (Op == rocjitsu::arith::Binary::MulDx9Zero) {
    if (rocjitsu::arith::is_exact(flags)) {
      return multiply_dx9_exact<Rdna3>(flags, exec_mask, uint32_t(mode), d, a, b);
    }
  }
  if constexpr (Op == rocjitsu::arith::Binary::Sub || Op == rocjitsu::arith::Binary::Subrev) {
    if (rocjitsu::arith::is_exact(flags)) {
      return subtract_exact<Op == rocjitsu::arith::Binary::Subrev, Rdna3>(flags, exec_mask,
                                                                          uint32_t(mode), d, a, b);
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::binary_x86_64_v3(Op, flags, exec_mask, mode, d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = rocjitsu::arith::alu_input(a[0][lane], mode),
          y = rocjitsu::arith::alu_input(b[0][lane], mode >> 1);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = rocjitsu::arith::flush_denorm_f32(x);
      y = rocjitsu::arith::flush_denorm_f32(y);
    }
    float value;
    if constexpr (Op == rocjitsu::arith::Binary::Add) {
      value = x + y;
    }
    if constexpr (Op == rocjitsu::arith::Binary::Sub) {
      value = x - y;
    }
    if constexpr (Op == rocjitsu::arith::Binary::Subrev) {
      value = y - x;
    }
    if constexpr (Op == rocjitsu::arith::Binary::MulDx9Zero) {
      value = (x == 0 || y == 0) ? 0.0f : x * y;
    }
    if constexpr (Op == rocjitsu::arith::Binary::Mul) {
      value = x * y;
    }

    value = rocjitsu::arith::as_float(
        rocjitsu::arith::scale_omod_f32_bits(rocjitsu::arith::as_bits(value), (mode >> 6) & 3,
                                             unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24)));
    const bool preserve_nan = (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
                              (rocjitsu::arith::as_bits(value) & 0x7fffffffU) > 0x7f800000U;
    value = rocjitsu::arith::alu_output(
        value, mode & ~(RJ_ARITH_ALU_OMOD_HALF | (preserve_nan ? RJ_ARITH_ALU_CLAMP : 0)));
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

// Evaluate FP32 binary arithmetic for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Binary Op, bool Rdna3 = false>
int binary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *state = nullptr) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = binary<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, state)) {
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
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error = binary<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                      &x, &y, state)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary<rocjitsu::arith::Binary::Add, true>(flags, exec_mask, mode, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  return binary<rocjitsu::arith::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary<rocjitsu::arith::Binary::Sub, true>(flags, exec_mask, mode, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  return binary<rocjitsu::arith::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary<rocjitsu::arith::Binary::Subrev, true>(flags, exec_mask, mode, d, a, b);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  return binary<rocjitsu::arith::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, mul_status_supported<true>(flags, mode)
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                 : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return binary<rocjitsu::arith::Binary::Mul, true>(flags, exec_mask, mode, d, a, b, trapsts);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, mul_status_supported<false>(flags, mode)
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                 : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return binary<rocjitsu::arith::Binary::Mul>(flags, exec_mask, mode, d, a, b, excp_flag_user);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_dx9_zero_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return binary<rocjitsu::arith::Binary::MulDx9Zero, true>(flags, exec_mask, mode, d, a, b);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_dx9_zero_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  return binary<rocjitsu::arith::Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary_wave64<rocjitsu::arith::Binary::Add, true>(flags, exec_mask, mode, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  return binary_wave64<rocjitsu::arith::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary_wave64<rocjitsu::arith::Binary::Sub, true>(flags, exec_mask, mode, d, a, b);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  return binary_wave64<rocjitsu::arith::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  return binary_wave64<rocjitsu::arith::Binary::Subrev, true>(flags, exec_mask, mode, d, a, b);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  return binary_wave64<rocjitsu::arith::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, mul_status_supported<true>(flags, mode)
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                 : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return binary_wave64<rocjitsu::arith::Binary::Mul, true>(flags, exec_mask, mode, d, a, b,
                                                           trapsts);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, mul_status_supported<false>(flags, mode)
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                 : RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return binary_wave64<rocjitsu::arith::Binary::Mul>(flags, exec_mask, mode, d, a, b,
                                                     excp_flag_user);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_dx9_zero_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return binary_wave64<rocjitsu::arith::Binary::MulDx9Zero, true>(flags, exec_mask, mode, d, a, b);
}

// Multiply the operands with DX9 zero-product rules for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_dx9_zero_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  return binary_wave64<rocjitsu::arith::Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
}
