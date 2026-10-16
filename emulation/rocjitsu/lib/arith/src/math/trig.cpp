// SPDX-License-Identifier: MIT

// FP32 sine/cosine, modifiers, and exact/loose policy selection.
// Owns public entry points and their validation/dispatch policy.

#include "math/trig.h"
#include "common/internal.h"
#include "math/trig_exceptions.h"
#include "math/trig_model.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Scale a finite trig result by an exact power of two with RNE underflow.
// Input magnitude is at most one; no multiplication can overflow.
template <bool Rdna3> uint32_t output(uint32_t bits, uint32_t mode, bool preserve_nan) {
  uint32_t magnitude = bits & 0x7fffffff;
  if constexpr (Rdna3) {
    // Borrowed apply_omod_f32: flush before scaling, retaining the sign when
    // halving a normal result into the subnormal range.
    int scale = (mode >> 6) & 3;
    if (scale && magnitude < 0x7f800000) {
      if (magnitude < 0x800000) {
        bits = 0;
      } else if (scale == 3 && magnitude < 0x1000000) {
        bits &= 0x80000000;
      } else {
        bits += scale == 3 ? -0x800000U : scale * 0x800000U;
      }
    }
    magnitude = bits & 0x7fffffff;
  } else if (magnitude < 0x7f800000) {
    int scale = (mode >> 6) & 3;
    if (scale == 3) {
      magnitude =
          magnitude < 0x1000000 ? (magnitude >> 1) + ((magnitude & 3) == 3) : magnitude - 0x800000;
    } else {
      for (int i = 0; i < scale; ++i) {
        magnitude = magnitude < 0x800000 ? magnitude * 2 : magnitude + 0x800000;
      }
    }
    bits = (bits & 0x80000000) | magnitude;
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) && (!preserve_nan || magnitude <= 0x7f800000)) {
    if ((bits >> 31) || magnitude > 0x7f800000) {
      return 0;
    }
    if (magnitude > 0x3f800000) {
      return 0x3f800000;
    }
  }
  if constexpr (!Rdna3) {
    if ((mode & RJ_ARITH_ALU_OMOD_HALF) && (bits & 0x7fffffff) < 0x800000) {
      return 0;
    }
  }
  return bits;
}

// Compute loose results with guest denormal and NaN policies, consuming all
// inputs before writing any destination.
#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate loose sine/cosine with guest denormal controls and output modifiers; preserve
// inactive lanes.
template <bool Rdna3, bool Cosine>
[[gnu::noinline]] void loose_policy(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                    const uint32_t *a) {
  uint32_t input[32], result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = a[lane];
    if (mode & RJ_ARITH_ALU_ABS_A) {
      bits &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      bits ^= 0x80000000;
    }
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fffffff) < 0x800000) {
      bits &= 0x80000000;
    }
    input[lane] = bits;
  }
  rocjitsu::arith::trig_x86_64_v3(Cosine, UINT32_MAX, 0, result, input);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = result[lane];
    if constexpr (Rdna3) {
      if ((input[lane] & 0x7fffffff) > 0x7f800000) {
        bits = input[lane] | (flags & RJ_ARITH_FP_IEEE_MODE ? 0x400000 : 0);
      }
    }
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fffffff) < 0x800000) {
      bits &= 0x80000000;
    }
    if ((exec_mask >> lane) & 1) {
      d[lane] = output<Rdna3>(bits, mode, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
  }
}

#endif

// Evaluate FP32 sine/cosine for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Cosine, bool Rdna3>
int trig(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, uint32_t *excp_flag_user) {
  const bool report = rocjitsu::arith::reports_fp_status(flags);
  if (report &&
      (Rdna3 ||
       (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) ||
       rocjitsu::arith::validate_semantics(flags))) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return trig<Cosine, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, excp_flag_user);
        });
  }
  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known, true,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_IEEE_MODE | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    if (flags & RJ_ARITH_FP_IEEE_MODE) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
    }
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3 && !rocjitsu::arith::is_exact(flags)) {
    if (Rdna3 ||
        (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      loose_policy<Rdna3, Cosine>(flags, exec_mask, mode, d[0], a[0]);
    } else {
      rocjitsu::arith::trig_x86_64_v3(Cosine, exec_mask, mode, d[0], a[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t exceptions = 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = a[0][lane];
    if (mode & RJ_ARITH_ALU_ABS_A) {
      bits &= 0x7fffffff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      bits ^= 0x80000000;
    }

    // The borrowed model encodes preservation, whereas API flags encode flushing; invert the two
    // denormal-control bits.
    unsigned denorm = 3 ^ (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS ? 1 : 0) ^
                      (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 2 : 0);
    uint32_t rounded = rocjitsu::arith::trig::evaluate(bits, Cosine, denorm,
                                                       !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
    result[lane] = output<Rdna3>(rounded, mode, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    if (report && ((exec_mask >> lane) & 1)) {
      exceptions |= rocjitsu::arith::trig::exceptions<32>(bits, rounded, Cosine, mode);
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

// Evaluate FP32 sine/cosine for Wave64. Reject invalid controls before changing outputs.
template <bool Cosine, bool Rdna3>
int trig_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, uint32_t *excp_flag_user) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = trig<Cosine, Rdna3>(flags, 0, mode, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool report = rocjitsu::arith::reports_fp_status(flags);

  // Delay the global-register write until both halves have consumed their inputs.
  uint32_t exceptions = 0;
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = trig<Cosine, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                        &source, report ? &exceptions : nullptr)) {
      return error;
    }
  }
  if (report) {
    *excp_flag_user |= exceptions;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute sine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sin_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute sine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sin_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig_wave64<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute sine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sin_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Compute sine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sin_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig_wave64<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Compute cosine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cos_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute cosine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cos_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig_wave64<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

// Compute cosine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cos_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Compute cosine for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cos_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig_wave64<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}
