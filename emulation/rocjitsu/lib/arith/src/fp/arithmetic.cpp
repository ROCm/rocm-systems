// SPDX-License-Identifier: MIT

// gfx12 Wave32 FP32 FMA with loose and empirical exact results.

#include "common/internal.h"
#include "fp/fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <cmath>

// Evaluate fused A*B+C for selected lanes; errors leave D and FP status unchanged.
template <int Lanes, bool Rdna3 = false>
static int fma(uint64_t flags, rocjitsu::arith::ExecMask<Lanes> exec_mask,
               uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
               const uint32_t *const *b, const uint32_t *const *c, uint32_t *excp_flag_user) {
  (void)excp_flag_user;
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, 0, true)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    if (rocjitsu::arith::is_exact(flags)) {
      result[lane] =
          rocjitsu::arith::fma_value_f32(a[0][lane], b[0][lane], c[0][lane], false, flags, Rdna3);
    } else {
      result[lane] = rocjitsu::arith::as_bits(std::fma(rocjitsu::arith::as_float(a[0][lane]),
                                                       rocjitsu::arith::as_float(b[0][lane]),
                                                       rocjitsu::arith::as_float(c[0][lane])));
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate F32 fused A*B+C for a Wave32 EXEC mask.
int rj_arith_v_fma_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  return fma<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Evaluate F32 fused A*B+C for a Wave64 EXEC mask.
int rj_arith_v_fma_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  return fma<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Evaluate gfx11 FP32 FMA for Wave32; required TRAPSTS reporting is unsupported.
int rj_arith_v_fma_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  return fma<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Evaluate gfx11 FP32 FMA for Wave64; required TRAPSTS reporting is unsupported.
int rj_arith_v_fma_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  return fma<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}
