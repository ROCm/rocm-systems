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

// Exact FMA, including source modifiers, output scaling and CLAMP.
// Sources may overlap the destination. Requires the default host FP environment.
template <bool Rdna3>
[[gnu::noinline]] int fma_exact(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::fma_exact_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                         c[0], Rdna3);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3) &&
      !(flags & RJ_ARITH_FP_ROUND_MASK)) {
    rocjitsu::arith::fma_exact_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0],
                                         c[0], Rdna3);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::fma_instruction_value(a[0][lane], b[0][lane], c[0][lane], flags,
                                                          instruction_flags, Rdna3);
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
template <bool Rdna3 = false>
int fma(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Rdna3) {
    instruction_flags = uint32_t(rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags));
  }

  if (rocjitsu::arith::is_exact(flags)) {
    return fma_exact<Rdna3>(flags, exec_mask, instruction_flags, d, a, b, c);
  }

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {

    rocjitsu::arith::fma_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);

    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {

    rocjitsu::arith::fma_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);

    return RJ_ARITH_SUCCESS;
  }
#endif

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(a[0][lane], instruction_flags, flags));
    float y = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(b[0][lane], instruction_flags >> 1, flags));
    float z = rocjitsu::arith::as_float(
        rocjitsu::arith::alu_input_bits(c[0][lane], instruction_flags >> 2, flags));

    float value = __builtin_fmaf(x, y, z);
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
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
template <bool Rdna3 = false>
int fma_with_dpp(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, uint32_t(instruction_flags) & ~rocjitsu::arith::kFma32InstructionFlags,
          rocjitsu::arith::kFmaGuestFpFlags | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t selected, const uint32_t *const *source) {
          return fma_with_dpp<Rdna3>(flags, selected, uint32_t(instruction_flags), d, source, b, c);
        });
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  return fma<Rdna3>(flags, exec_mask, uint32_t(instruction_flags), d, a, b, c);
}

// Evaluate FP32 fused multiply-add for Wave64. Reject invalid controls before changing
// outputs.
template <bool Rdna3 = false>
int fma_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  int error = fma_with_dpp<Rdna3>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr);
  if (error || !exec_mask) {
    return error;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    uint32_t half_exec_mask = uint32_t(exec_mask >> (half * 32));
    error = fma_with_dpp<Rdna3>(flags, half_exec_mask, half_instruction_flags, &dest, &x, &y, &z);
    if (error) {
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

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return fma_wave64<>(flags, exec_mask, instruction_flags, d, a, b, c);
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
  return fma_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
