// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP8/BF8 DOT4 and DOT8 with FP32 accumulators.
// Owns public entry points and their validation/dispatch policy.

#include "common/float_formats.h"
#include "common/internal.h"
#include "common/simd.h"
#include "dot/dot.h"
#include "dot/operand_halves.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Accumulate packed FP8/BF8 dot products for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Bf8A, bool Bf8B>
int dot(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, modifiers, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return dot<Bf8A, Bf8B>(flags, exec_mask, uint32_t(modifiers), d, source, b, c);
        });
  }
  const uint32_t known =
      rocjitsu::arith::kDotHalfSelection | RJ_ARITH_DOT_NEG_C | RJ_ARITH_DOT_ABS_C;
  if (int error = rocjitsu::arith::validate(flags, modifiers & ~known)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  if (modifiers & rocjitsu::arith::kDotHalfSelection) {
    return rocjitsu::arith::with_dot_halves(
        modifiers, a, b,
        [&](uint32_t selected, const uint32_t *const *av, const uint32_t *const *bv) {
          return dot<Bf8A, Bf8B>(flags, exec_mask, selected, d, av, bv, c);
        });
  }
  if (rocjitsu::arith::is_exact(flags)) {
    using namespace rocjitsu::arith::gfx12_dot_detail;
    constexpr auto fa = Bf8A ? InputFormat::Bf8 : InputFormat::Fp8;
    constexpr auto fb = Bf8B ? InputFormat::Bf8 : InputFormat::Fp8;
    uint32_t result[32];
    for (int lane = 0; lane < 32; ++lane) {
      uint8_t av[4], bv[4];
      for (int byte = 0; byte < 4; ++byte) {
        av[byte] = uint8_t(a[0][lane] >> (8 * byte));
        bv[byte] = uint8_t(b[0][lane] >> (8 * byte));
      }
      uint32_t acc = c[0][lane];
      if (modifiers & RJ_ARITH_DOT_ABS_C) {
        acc &= 0x7fffffffU;
      }
      if (modifiers & RJ_ARITH_DOT_NEG_C) {
        acc ^= 0x80000000U;
      }
      result[lane] = dot_bits<fa, fb, false>(av, bv, acc);
    }
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[0][lane] = result[lane];
      }
    }
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::fp8_dot_x86_64_v4(Bf8A, Bf8B, exec_mask, modifiers, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::fp8_dot_x86_64_v3(Bf8A, Bf8B, exec_mask, modifiers, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t acc_bits = c[0][lane];
    if (modifiers & RJ_ARITH_DOT_ABS_C) {
      acc_bits &= 0x7fffffff;
    }
    if (modifiers & RJ_ARITH_DOT_NEG_C) {
      acc_bits ^= 0x80000000;
    }
    float acc = rocjitsu::arith::as_float(acc_bits);
    for (int shift = 0; shift < 32; shift += 8) {
      acc = __builtin_fmaf(rocjitsu::arith::fp8_to_float<Bf8A>(uint8_t(a[0][lane] >> shift)),
                           rocjitsu::arith::fp8_to_float<Bf8B>(uint8_t(b[0][lane] >> shift)), acc);
    }
    result[lane] = rocjitsu::arith::as_bits(acc);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Accumulate packed FP8/BF8 dot products for Wave64. Reject invalid controls before changing
// outputs.
template <bool Bf8A, bool Bf8B>
int dot_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = dot<Bf8A, Bf8B>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = dot<Bf8A, Bf8B>(flags, uint32_t(exec_mask >> (half * 32)),
                                    half_instruction_flags, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_fp8_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<false, false>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_fp8_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<false, true>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_bf8_fp8_w32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<true, false>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_bf8_bf8_w32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<true, true>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_fp8_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<false, false>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_fp8_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<false, true>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_bf8_fp8_w64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<true, false>(flags, exec_mask, modifiers, d, a, b, c);
}

// Accumulate four packed products for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_dot4_f32_bf8_bf8_w64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<true, true>(flags, exec_mask, modifiers, d, a, b, c);
}
