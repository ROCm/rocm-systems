// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed FP16/FP32 FMA with FP32 or selected-halfword output.
// Owns public entry points and their validation/dispatch policy.

#include "fp/mixed_fma.h"
#include "common/alu.h"
#include "common/internal.h"
#include "fp/fma_scalar.h"
#include "fp/mixed_fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

static const uint64_t fp_flags =
    rocjitsu::arith::kFmaGuestFpFlags | RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK;

// Evaluate mixed FP16/FP32 fused multiply-add for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::MixedFma Dst, bool Rdna3 = false>
int mixed_fma(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
              const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return mixed_fma<Dst, Rdna3>(flags, exec_mask,
                                                                       uint32_t(instruction_flags),
                                                                       d, source, b, c);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                         RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B |
                         RJ_ARITH_MIX_F16_C;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known,
                                            fp_flags | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if ((flags & RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) == RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if ((Rdna3 || Dst == rocjitsu::arith::MixedFma::Float) &&
      rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool exact = rocjitsu::arith::is_exact(flags);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::mixed_fma_exact_x86_64_v4(Dst, Rdna3, flags, exec_mask,
                                               uint32_t(instruction_flags), d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3) &&
      !(flags & RJ_ARITH_FP_ROUND_MASK)) {
    if constexpr (Dst == rocjitsu::arith::MixedFma::Float) {
      rocjitsu::arith::mixed_fma_float_exact_x86_64_v3(
          Rdna3, flags, exec_mask, uint32_t(instruction_flags), d[0], a[0], b[0], c[0]);
    } else {
      // FP32 products are exact in FP64. The existing compensated sum keeps
      // its residual when rounding to odd, avoiding double rounding to FP16.
      rocjitsu::arith::mixed_fma_half_guest_x86_64_v3(Dst == rocjitsu::arith::MixedFma::High, flags,
                                                      exec_mask, uint32_t(instruction_flags), d[0],
                                                      a[0], b[0], c[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
  if (!exact && rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Dst == rocjitsu::arith::MixedFma::Float) {
      if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                   RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK)) {
        rocjitsu::arith::mixed_fma_float_guest_x86_64_v3(flags, exec_mask, instruction_flags, d[0],
                                                         a[0], b[0], c[0]);
      } else {
        rocjitsu::arith::mixed_fma_float_x86_64_v3(exec_mask, instruction_flags, d[0], a[0], b[0],
                                                   c[0]);
      }
    } else if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                        RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK)) {
      rocjitsu::arith::mixed_fma_half_guest_x86_64_v3(Dst == rocjitsu::arith::MixedFma::High, flags,
                                                      exec_mask, instruction_flags, d[0], a[0],
                                                      b[0], c[0]);
    } else {
      rocjitsu::arith::mixed_fma_half_x86_64_v3(Dst == rocjitsu::arith::MixedFma::High,
                                                flags & RJ_ARITH_FP16_OVFL, exec_mask,
                                                instruction_flags, d[0], a[0], b[0], c[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = rocjitsu::arith::mixed_fma_input(
                 a[0][lane], instruction_flags,
                 rocjitsu::arith::mixed_fma_flush_input(flags, instruction_flags)),
             y = rocjitsu::arith::mixed_fma_input(
                 b[0][lane], instruction_flags >> 1,
                 rocjitsu::arith::mixed_fma_flush_input(flags, instruction_flags >> 1)),
             z = rocjitsu::arith::mixed_fma_input(
                 c[0][lane], instruction_flags >> 2,
                 rocjitsu::arith::mixed_fma_flush_input(flags, instruction_flags >> 2));
    if constexpr (Dst == rocjitsu::arith::MixedFma::Float) {
      if (exact) {
        result[lane] = rocjitsu::arith::fma_value_f32(x, y, z, flags, Rdna3);
        if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
          result[lane] = rocjitsu::arith::fma_clamp_f32(result[lane], flags);
        }
        continue;
      }
      float value = __builtin_fmaf(rocjitsu::arith::as_float(x), rocjitsu::arith::as_float(y),
                                   rocjitsu::arith::as_float(z));
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = rocjitsu::arith::flush_denorm_f32(value);
      }
      if ((instruction_flags & RJ_ARITH_ALU_CLAMP) &&
          !((flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
            (rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000)) {
        value = !(value > 0.0f) ? 0.0f : value > 1.0f ? 1.0f : value;
      }
      result[lane] = rocjitsu::arith::as_bits(value);
    } else {
      constexpr unsigned shift = Dst == rocjitsu::arith::MixedFma::High ? 16 : 0;
      uint16_t half = rocjitsu::arith::mixed_fma_half_value(
          x, y, z, instruction_flags & RJ_ARITH_ALU_CLAMP, flags);
      result[lane] = (d[0][lane] & ~(uint32_t(65535) << shift)) | (uint32_t(half) << shift);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate mixed FP16/FP32 fused multiply-add for Wave64. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::MixedFma Dst, bool Rdna3 = false>
int mixed_fma_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          mixed_fma<Dst, Rdna3>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    if (int error = mixed_fma<Dst, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)),
                                          half_instruction_flags, &dest, &x, &y, &z)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mix_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d, a, b,
                                                     c);
}

// RX 9070 MIXLO/MIXHI do not generate exception flags, including invalid
// arithmetic and overflow. A requested update therefore leaves the register
// unchanged, including every FP32/FP16 denormal setting in hardware captures.
int rj_arith_v_fma_mixlo_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute mixed FMA into the high FP16 half. Return errors before changing outputs.
// Like MIXLO above, requested FP-status reporting leaves the register unchanged.
int rj_arith_v_fma_mixhi_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d, a, b,
                                                    c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mix_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d,
                                                            a, b, c);
}

// RX 9070 MIXLO/MIXHI do not generate exception flags, including invalid
// arithmetic and overflow. A requested update therefore leaves the register
// unchanged, including every FP32/FP16 denormal setting in hardware captures.
int rj_arith_v_fma_mixlo_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Compute mixed FMA into the high FP16 half. Return errors before changing outputs.
// Like MIXLO above, requested FP-status reporting leaves the register unchanged.
int rj_arith_v_fma_mixhi_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mix_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::Float, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mixlo_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::Low, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mixhi_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma<rocjitsu::arith::MixedFma::High, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mix_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::Float, true>(flags, exec_mask,
                                                                  instruction_flags, d, a, b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mixlo_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::Low, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Compute fused A*B+C with mixed FP16/FP32 operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_mixhi_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return mixed_fma_wave64<rocjitsu::arith::MixedFma::High, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}
