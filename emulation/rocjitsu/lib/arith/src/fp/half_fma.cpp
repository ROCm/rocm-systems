// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx11/gfx12 FP16 FMA with guest rounding and optional FP status.
// Owns public entry points and their validation/dispatch policy.

#include "common/alu.h"
#include "common/internal.h"
#include "fp/half_fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate FP16 fused multiply-add for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  bool report = rocjitsu::arith::reports_fp_status(flags);
  if (instruction_flags >> 32) {

    return rocjitsu::arith::execute_dpp(
        exec_mask, instruction_flags, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return run<Rdna3>(flags, exec_mask, uint32_t(instruction_flags), d, source, b, c,
                            excp_flag_user);
        });
  }
  const uint32_t known = rocjitsu::arith::kFma16InstructionFlags;
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
  int a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = instruction_flags & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t x = uint16_t(a[0][lane] >> a_shift), y = uint16_t(b[0][lane] >> b_shift);
    uint16_t z = uint16_t(c[0][lane] >> c_shift);

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
template <bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Rdna3>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr,
                             excp_flag_user)) {
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
    const uint32_t *z = c[0] + half * 32;
    if (int error = run<Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                               &dest, &x, &y, &z, report ? &exceptions : nullptr)) {
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

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}
