// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 sine/cosine with guest input/output modifiers.
// Owns public entry points and their validation/dispatch policy.

#include "common/float_formats.h"
#include "common/internal.h"
#include "math/trig.h"
#include "math/trig_exceptions.h"
#include "math/trig_model.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Apply output modifiers to an already rounded FP16 trig result. Active OMOD
// flushes tiny inputs before scaling and tiny outputs after RNE narrowing.
uint16_t output(uint16_t bits, uint32_t mode, bool preserve_nan) {
  uint32_t magnitude = bits & 0x7fff;
  unsigned scale = (mode >> 6) & 3;
  if (scale && magnitude < 0x7c00) {
    if (magnitude < 0x400) {
      bits = 0;
    } else {
      if (scale == 3) {
        magnitude =
            magnitude < 0x800 ? (magnitude >> 1) + ((magnitude & 3) == 3) : magnitude - 0x400;
      } else {
        magnitude += scale * 0x400;
      }
      bits = magnitude < 0x400 ? 0 : uint16_t((bits & 0x8000) | magnitude);
    }
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) && (!preserve_nan || (bits & 0x7fff) <= 0x7c00)) {
    if ((bits & 0x8000) || (bits & 0x7fff) > 0x7c00) {
      return 0;
    }
    if (bits > 0x3c00) {
      return 0x3c00;
    }
  }
  return bits;
}

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Compute loose results with guest half-precision policies. Consume all inputs
// before writing selected destination halves; preserve the other halves.
template <bool Rdna3, bool Cosine>
[[gnu::noinline]] void loose_policy(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                    const uint32_t *a) {
  uint32_t input[32], result[32]{};
  unsigned a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[lane] >> a_shift);
    if (mode & RJ_ARITH_ALU_ABS_A) {
      bits &= 0x7fff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      bits ^= 0x8000;
    }
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fff) < 0x400) {
      bits &= 0x8000;
    }
    input[lane] = bits;
  }
  rocjitsu::arith::half_trig_x86_64_v3(Cosine, UINT32_MAX, 0, result, input);
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(result[lane]);
    if constexpr (Rdna3) {
      if ((input[lane] & 0x7fff) > 0x7c00) {
        bits = uint16_t(input[lane] | (flags & RJ_ARITH_FP_IEEE_MODE ? 0x200 : 0));
      }
    }
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fff) < 0x400) {
      bits &= 0x8000;
    }
    if ((exec_mask >> lane) & 1) {
      d[lane] = (d[lane] & ~(0xffffU << d_shift)) |
                (uint32_t(output(bits, mode, flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) << d_shift);
    }
  }
}
#endif

// Evaluate FP16 sine/cosine for the selected lanes. Reject invalid controls before changing
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
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_IEEE_MODE | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
    }
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3) &&
      !rocjitsu::arith::is_exact(flags)) {
    if (Rdna3 ||
        (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      loose_policy<Rdna3, Cosine>(flags, exec_mask, mode, d[0], a[0]);
    } else {
      rocjitsu::arith::half_trig_x86_64_v3(Cosine, exec_mask, mode, d[0], a[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  const int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t half = uint16_t(a[0][lane] >> a_shift);
    if (mode & RJ_ARITH_ALU_ABS_A) {
      half &= 0x7fff;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      half ^= 0x8000;
    }
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (half & 0x7fff) < 0x400) {
      half &= 0x8000;
    }

    // Evaluate the inherited FP32 model after explicit FP16 input flushing, then round to FP16
    // before output modifiers.
    uint32_t bits = rocjitsu::arith::trig::evaluate(
        rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(half)), Cosine, 3,
        !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
    uint16_t rounded = rocjitsu::arith::float_to_f16(rocjitsu::arith::as_float(bits));
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (rounded & 0x7fff) < 0x400) {
      rounded &= 0x8000;
    }
    result[lane] = output(rounded, mode, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    if (report && ((exec_mask >> lane) & 1)) {
      exceptions |= rocjitsu::arith::trig::exceptions<16>(half, rounded, Cosine, mode);
    }
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

// Evaluate FP16 sine/cosine for Wave64. Reject invalid controls before changing outputs.
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
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error =
            trig<Cosine, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                &dest, &source, report ? &exceptions : nullptr)) {
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
int rj_arith_v_sin_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
int rj_arith_v_sin_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
int rj_arith_v_sin_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
int rj_arith_v_sin_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
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
int rj_arith_v_cos_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
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
int rj_arith_v_cos_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
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
int rj_arith_v_cos_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
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
int rj_arith_v_cos_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return trig_wave64<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}
