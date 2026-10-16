// SPDX-License-Identifier: MIT

// Exponent extraction from FP16, FP32, and FP64 encodings.
// Owns public entry points and their validation/dispatch policy.

#include "fp/frexp_exp.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Extract floating-point exponents for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Fp64>
int frexp_exp(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
              const uint32_t *const *a) {
  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~known, !Fp64,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }

  // Sign modifiers do not affect the exponent. Integer results ignore OMOD;
  // the exponent always fits in int32_t, so CLAMP does not change the result.
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::frexp_exp_x86_64_v3(Fp64, exec_mask, d[0], a,
                                         flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    // Match rocjitsu's frexp exponent model, including zero for NaNs/infinities.
    int exponent = 0;
    if constexpr (Fp64) {
      uint64_t magnitude = (uint64_t(a[0][lane]) | (uint64_t(a[1][lane]) << 32)) & INT64_MAX;
      unsigned field = unsigned(magnitude >> 52);
      if (field && field != 2047) {
        exponent = int(field) - 1022;
      } else if (!field && magnitude) {
        // The leading bit is 63-clz; a subnormal exponent is that position-1073.
        exponent = -1010 - int(rocjitsu::arith::bit_count_zero<false>(magnitude));
      }
    } else {
      uint32_t magnitude = a[0][lane] & 0x7fffffffu;
      unsigned field = magnitude >> 23;
      if (field && field != 255) {
        exponent = int(field) - 126;
      } else if (!field && magnitude && !(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        // rocjitsu's frexp_f32: exponent = -125 - (countl_zero(magnitude) - 8).
        exponent = -117 - int(rocjitsu::arith::bit_count_zero<false>(magnitude));
      }
    }
    result[lane] = uint32_t(exponent);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Keep DPP's routing buffers off the ordinary instruction's call path.
[[gnu::noinline]] int frexp_exp_f32_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  return rocjitsu::arith::execute_dpp(
      flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
        return frexp_exp<false>(flags, exec_mask, uint32_t(mode), d, source);
      });
}

} // namespace

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // Both architectures use the same numeric model; RDNA3 status is not yet verified.
  // DX10_CLAMP only affects floating-point outputs, not this integer exponent.
  return rj_arith_v_frexp_exp_i32_f32_w32_gfx12(flags & ~RJ_ARITH_FP_DX10_CLAMP_OFF, exec_mask,
                                                mode, d, a, nullptr);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  const auto semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (rocjitsu::arith::reports_fp_status(flags) && semantics != RJ_ARITH_SEMANTICS_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return frexp_exp_f32_dpp(flags, exec_mask, mode, d, a);
  }
  return frexp_exp<false>(flags, exec_mask, mode, d, a);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return frexp_exp<true>(flags, exec_mask, mode, d, a);
}

namespace {

// Extract floating-point exponents for Wave64. Reject invalid controls before changing
// outputs.
template <bool Fp64, bool Rdna4 = false>
int frexp_exp_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, uint32_t *excp_flag_user) {
  const auto instruction = Fp64    ? rj_arith_v_frexp_exp_i32_f64_w32_gfx12
                           : Rdna4 ? rj_arith_v_frexp_exp_i32_f32_w32_gfx12
                                   : rj_arith_v_frexp_exp_i32_f32_w32;

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, excp_flag_user)) {
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
    const uint32_t *source[Fp64 ? 2 : 1];
    for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
      source[reg] = a[reg] + half * 32;
    }
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, source,
                                excp_flag_user)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return frexp_exp_wave64<false>(flags, exec_mask, mode, d, a, trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return frexp_exp_wave64<false, true>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return frexp_exp_wave64<true>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_frexp_exp_i32_f64_w32_gfx12(flags, exec_mask, mode, d, a, trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_exp_i32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_frexp_exp_i32_f64_w64_gfx12(flags, exec_mask, mode, d, a, trapsts);
}
