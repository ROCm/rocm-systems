// SPDX-License-Identifier: MIT

// FP16/BF16 DOT2 with packed 16-bit accumulators and results.
// Owns public entry points and their validation/dispatch policy.
// True16 DOT2 association and denormal policy follow rocjitsu shared/fp_mode.h.

#include "common/float_formats.h"
#include "common/internal.h"
#include "common/simd.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Widen one FP16/BF16 encoding and apply its ABS/NEG operand modifiers.
template <bool Bf16> float input(uint16_t bits, uint32_t mode) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits &= 0x7fff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits ^= 0x8000;
  }
  if constexpr (Bf16) {
    if (!(bits & 0x7f80)) {
      bits &= 0x8000;
    }
    return rocjitsu::arith::bf16_to_float(bits);
  } else {
    return rocjitsu::arith::f16_to_float(bits);
  }
}

// Accumulate DOT2 into a 16-bit result for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Bf16>
int dot(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return dot<Bf16>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  // Six ABS/NEG bits, plus C and D half selectors.
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(63U | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D), false,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_IEEE_MODE)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::half_dot_x86_64_v4(Bf16, bool(flags & RJ_ARITH_FP16_OVFL), exec_mask, mode,
                                        d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::half_dot_x86_64_v3(Bf16, bool(flags & RJ_ARITH_FP16_OVFL), exec_mask, mode,
                                        d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  int c_shift = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float p0 =
        input<Bf16>(uint16_t(a[0][lane]), mode) * input<Bf16>(uint16_t(b[0][lane]), mode >> 1);
    float p1 = input<Bf16>(uint16_t(a[0][lane] >> 16), mode) *
               input<Bf16>(uint16_t(b[0][lane] >> 16), mode >> 1);
    float accumulator = input<Bf16>(uint16_t(c[0][lane] >> c_shift), mode >> 2);
    float value = (p0 + p1) + accumulator;
    if constexpr (Bf16) {
      result[lane] = rocjitsu::arith::float_to_bf16(value);
      if (!(result[lane] & 0x7f80)) {
        result[lane] &= 0x8000;
      }
    } else {
      result[lane] = rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Accumulate DOT2 into a 16-bit result for Wave64. Reject invalid controls before changing
// outputs.
template <bool Bf16>
int dot_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = dot<Bf16>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error =
            dot<Bf16>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<false>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<false>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_bf16_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<true>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_bf16_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<true>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<false>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<false>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_bf16_bf16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<true>(flags, exec_mask, mode, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_bf16_bf16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<true>(flags, exec_mask, mode, d, a, b, c);
}
