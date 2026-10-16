// SPDX-License-Identifier: MIT

// Packed FP16 FMA/FMAC with independent low/high operand selectors.
// Owns public entry points and their validation/dispatch policy.

#include "fp/packed_fma.h"
#include "common/exact_arithmetic.h"
#include "common/internal.h"
#include "common/packed_alu.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

static const uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                 RJ_ARITH_FP_ROUND_MASK;

// Evaluate paired FP16 fused multiply-adds for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Rdna3, bool Accumulate = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  bool report = rocjitsu::arith::reports_fp_status(flags);
  constexpr uint32_t allowed = Accumulate ? RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
                                                RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B |
                                                RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_HALF
                                          : 0x1fffU;
  if (int error = rocjitsu::arith::validate(flags, mode & ~allowed, true,
                                            fp_flags | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  uint32_t modes[2];
  if constexpr (Accumulate) {
    // Packed F16 ignores OMOD on both RDNA3 and RDNA4. ABS/NEG apply to
    // both A/B components; the destination accumulator is unmodified.
    modes[0] = mode & ~RJ_ARITH_ALU_OMOD_HALF;
    modes[1] = modes[0] | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C;
  } else {
    modes[0] = rocjitsu::arith::packed_half_mode(mode, false);
    modes[1] = rocjitsu::arith::packed_half_mode(mode, true);
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4 &&
      rocjitsu::arith::packed_fma_exact_x86_64_v4(flags, exec_mask, modes, d[0], a[0], b[0], c[0],
                                                  excp_flag_user, Rdna3)) {
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK)) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3 &&
      rocjitsu::arith::packed_fma_exact_x86_64_v3(flags, exec_mask, modes, d[0], a[0], b[0], c[0],
                                                  excp_flag_user, Rdna3)) {
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  bool exact = rocjitsu::arith::is_exact(flags);
  const bool clamp = mode & (Accumulate ? RJ_ARITH_ALU_CLAMP : RJ_ARITH_PK_CLAMP);
  if (!exact && !(flags & RJ_ARITH_FP_ROUND_MASK) &&
      !(clamp && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) &&
      (flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::packed_fma_x86_64_v3<Accumulate>(flags, exec_mask, mode, d[0], a[0], b[0],
                                                      c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t exceptions = 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < 2; ++half) {
      uint32_t m = modes[half];
      uint16_t x = uint16_t(a[0][lane] >> (m & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      uint16_t y = uint16_t(b[0][lane] >> (m & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
      uint16_t z = uint16_t(c[0][lane] >> (m & RJ_ARITH_ALU_HIGH_C ? 16 : 0));
      result[lane] |=
          uint32_t(rocjitsu::arith::half_fma_value<Rdna3>(
              x, y, z, m, flags, report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr))
          << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate paired FP16 fused multiply-adds for Wave64. Reject invalid controls before changing
// outputs.
template <bool Rdna3, bool Accumulate = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Rdna3, Accumulate>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr,
                                         excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    if (int error = run<Rdna3, Accumulate>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest,
                                           &x, &y, &z, report ? &exceptions : nullptr)) {
      return error;
    }
  }
  // Delay the scalar output so it may alias any source, including the upper half.
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fma_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

// Accumulate fused A*B into D with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fmac_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, false)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false, true>(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

// Compute fused A*B+C with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fma_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

// Accumulate fused A*B into D with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fmac_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, false)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false, true>(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

// Compute fused A*B+C with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fma_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

// Accumulate fused A*B into D with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fmac_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, true)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}

// Compute fused A*B+C with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fma_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

// Accumulate fused A*B into D with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_fmac_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags) &&
      rocjitsu::arith::half_fma_reporting_unsupported(flags, true)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}
