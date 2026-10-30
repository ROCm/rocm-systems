// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reciprocal-with-flags entry points and optional guest FP exception reporting.
// Owns public entry points and their validation/dispatch policy.

// Reciprocal stages follow rocjitsu's transcendental and fp_mode helpers.
// GFX1201 corrections to its exception classifier: flushed subnormals also
// raise INT_DIV0, while CLAMP suppresses a new cause (not an existing flag).

#include "math/rcp_iflag.h"
#include "common/internal.h"
#include "math/unary_exact.h"
#include "math/x86_64/rcp_exact.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Compute reciprocal and optional guest causes for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Rdna3>
int rcp_iflag(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
              const uint32_t *const *a, uint32_t *excp_flag_user) {
  if ((mode >> 32) &&
      (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode))) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(
          flags, uint32_t(mode) & ~known,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_IEEE_MODE | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    // RDNA3 OMOD is disabled with preserved output denormals or IEEE mode.
    // The reciprocal's forced flushing does not change that guest-mode test.
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
    }
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (mode >> 32) {
    uint32_t permuted[32];
    const uint32_t *source = permuted;
    if (mode & RJ_ARITH_DPP8) {
      rocjitsu::arith::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
    } else {
      exec_mask = rocjitsu::arith::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
    }
    return rcp_iflag<Rdna3>(flags, exec_mask, uint32_t(mode), d, &source, excp_flag_user);
  }
  const uint32_t prior_exceptions = rocjitsu::arith::reports_fp_status(flags) ? *excp_flag_user : 0;
  if (rocjitsu::arith::is_exact(flags)) {
    // Snapshot the existing INT_DIV0 cause before D can overwrite an input.
    uint32_t generated = 0;
    if (rocjitsu::arith::reports_fp_status(flags) && (Rdna3 || !(mode & RJ_ARITH_ALU_CLAMP))) {
      uint32_t zeros = 0;

      // RDNA3 inherits rocjitsu's classify_rcp_iflag_f32_exceptions:
      // signed zero raises INT_DIV0 regardless of CLAMP. RDNA4 instead uses
      // the captured flushed-subnormal and CLAMP-suppression policy.
      for (int lane = 0; lane < 32; ++lane) {
        zeros |= uint32_t(!(a[0][lane] & (Rdna3 ? 0x7fffffffU : 0x7f800000U))) << lane;
      }
      generated = zeros & exec_mask ? RJ_ARITH_EXCEPTION_INT_DIV0 : 0;
    }
    // RCP_IFLAG always quiets NaNs, while OMOD gating was applied above.
    // Reuse the same integer reciprocal model without reapplying that gating.
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::rcp_exact_x86_64_v4(flags, exec_mask, uint32_t(mode), d[0], a[0], true);
      if (rocjitsu::arith::reports_fp_status(flags)) {
        *excp_flag_user = prior_exceptions | generated;
      }
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::rcp_exact_x86_64_v3(flags, exec_mask, uint32_t(mode), d[0], a[0], true);
      if (rocjitsu::arith::reports_fp_status(flags)) {
        *excp_flag_user = prior_exceptions | generated;
      }
      return RJ_ARITH_SUCCESS;
    }
#endif
    uint32_t result[32];
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = rocjitsu::arith::unary_exact_value<rocjitsu::arith::Unary::Rcp>(
          flags, uint32_t(mode), a[0][lane], false);
    }
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[0][lane] = result[lane];
      }
    }
    if (rocjitsu::arith::reports_fp_status(flags)) {
      *excp_flag_user = prior_exceptions | generated;
    }
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::rcp_iflag_x86_64_v4(flags, exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::rcp_iflag_x86_64_v3(flags, exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  unsigned omod = (mode >> 6) & 3;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    bool tiny = !(raw & 0x7f800000);
    if (mode & RJ_ARITH_ALU_ABS_A) {
      raw &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      raw ^= 0x80000000;
    }
    if (tiny) {
      raw &= 0x80000000;
    }
    float value = 1.f / rocjitsu::arith::as_float(raw);
    uint32_t bits = rocjitsu::arith::as_bits(value);
    if (!(bits & 0x7f800000)) {
      bits &= 0x80000000;
    }
    value = rocjitsu::arith::as_float(bits);
    if (omod) {
      if (value == 0) {
        value = 0;
      }
      value *= omod == 1 ? 2 : omod == 2 ? 4 : 0.5f;
    }
    if ((mode & RJ_ARITH_ALU_CLAMP) &&
        !((flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
          (rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000)) {
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    }
    bits = rocjitsu::arith::as_bits(value);
    if (!(bits & 0x7f800000)) {
      bits &= 0x80000000;
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

// Compute reciprocal and optional guest causes for Wave64. Reject invalid controls before
// changing outputs.
template <bool Rdna3>
int rcp_iflag_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, uint32_t *excp_flag_user) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = rcp_iflag<Rdna3>(flags, 0, mode, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Rdna3) {
    if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_LOOSE) {
      excp_flag_user = nullptr;
    }
  }

  // Preserve the register's pre-instruction value even when it aliases D.
  uint32_t exceptions = rocjitsu::arith::reports_fp_status(flags) ? *excp_flag_user : 0;
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = rcp_iflag<Rdna3>(
            flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags, &dest, &source,
            rocjitsu::arith::reports_fp_status(flags) ? &exceptions : nullptr)) {
      return error;
    }
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    *excp_flag_user = exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_iflag_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rcp_iflag<true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_iflag_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rcp_iflag_wave64<true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_iflag_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rcp_iflag<false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_iflag_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rcp_iflag_wave64<false>(flags, exec_mask, mode, d, a, excp_flag_user);
}
