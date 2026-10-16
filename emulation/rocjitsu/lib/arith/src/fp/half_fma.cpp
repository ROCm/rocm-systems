// SPDX-License-Identifier: MIT

// gfx12 Wave32 FP16 fused multiply-add, with explicit EXEC and whole-VGPR aliases.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

// Evaluate the low halfword and preserve the high half and inactive lanes.
// Modes 0..2 do not access FP status; mode 3 accumulates active-lane causes.
template <int Lanes, bool Rdna3 = false>
static int fma(uint64_t flags, rocjitsu::arith::ExecMask<Lanes> exec_mask,
               uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
               const uint32_t *const *b, const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (instruction_flags) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, 0, true)) {
    return error;
  }
  if (Rdna3 && rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  uint16_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    if (rocjitsu::arith::is_exact(flags)) {
      result[lane] = rocjitsu::arith::half_fma_value<Rdna3>(
          uint16_t(a[0][lane]), uint16_t(b[0][lane]), uint16_t(c[0][lane]), 0, flags,
          report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr);
    } else {
      float x = rocjitsu::arith::f16_to_float(uint16_t(a[0][lane]));
      float y = rocjitsu::arith::f16_to_float(uint16_t(b[0][lane]));
      float z = rocjitsu::arith::f16_to_float(uint16_t(c[0][lane]));
      result[lane] = rocjitsu::arith::float_to_f16(std::fma(x, y, z), flags & RJ_ARITH_FP16_OVFL);
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & 0xffff0000U) | result[lane];
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate F16 fused A*B+C for a Wave32 EXEC mask.
int rj_arith_v_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  return fma<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Evaluate F16 fused A*B+C for a Wave64 EXEC mask.
int rj_arith_v_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 uint32_t *excp_flag_user) {
  return fma<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

// Evaluate gfx11 FP16 FMA for Wave32; required TRAPSTS reporting is unsupported.
int rj_arith_v_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  return fma<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

// Evaluate gfx11 FP16 FMA for Wave64; required TRAPSTS reporting is unsupported.
int rj_arith_v_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *trapsts) {
  return fma<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}
