// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Legacy half MAD uses a rounded FP32 intermediate before FP16 narrowing.
// Gfx90a Compatibility instead rounds both the product and the sum to half.

#include "fp/half_mad.h"
#include "common/alu.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/float_output.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace {

// Validate the loose-only model before reading inputs or changing outputs.
int validate_mad(uint64_t &flags, uint64_t mode) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~uint64_t(rocjitsu::arith::kFma16InstructionFlags),
          rocjitsu::arith::kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  return RJ_ARITH_SUCCESS;
}

// Select and modify a raw half without changing its NaN payload or quiet bit.
uint16_t input_half(uint32_t word, uint32_t mode) {
  uint16_t bits = uint16_t(word >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits &= 0x7fff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits ^= 0x8000;
  }
  return bits;
}

// Evaluate active lanes, preserving the unselected destination half and allowing register aliases.
void mad(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::half_mad_x86_64_v4(flags, exec_mask, mode, d, a, b, c);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_mad_x86_64_v3(flags, exec_mask, mode, d, a, b, c);
    return;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = rocjitsu::arith::f16_to_float(input_half(a[lane], mode));
    float y = rocjitsu::arith::f16_to_float(input_half(b[lane], mode >> 1));
    float z = rocjitsu::arith::f16_to_float(input_half(c[lane], mode >> 2));
    // Half products fit exactly in FP32. Keep the subsequent FP32 rounding:
    // fused rounding directly to half would change inherited MAD results.
    float value = x * y + z;
    uint16_t half = rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL);
    result[lane] = rocjitsu::arith::selection_output<16>(half, mode, flags);
  }
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[lane] = (d[lane] & ~(0xffffU << shift)) | (result[lane] << shift);
    }
  }
}

// Evaluate gfx90a's two half roundings, with sources staged before any aliased writes.
void separately_rounded_mad(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  alignas(64) uint32_t x[32], y[32], z[32], product[32] = {}, result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    x[lane] = input_half(a[lane], mode);
    y[lane] = input_half(b[lane], mode >> 1);
    z[lane] = input_half(c[lane], mode >> 2);
  }
  // CDNA2 flushes at both arithmetic steps regardless of guest denormal controls.
  // Keep the guest rounding, NaN quieting and FP16_OVFL policy in the shared SIMD kernels.
  const uint64_t arithmetic = (flags & ~RJ_ARITH_SEMANTICS_MASK) |
                              RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  const uint32_t *px = x, *py = y, *pz = z, *pp = product;
  uint32_t *pd = product, *pr = result;
  rj_arith_v_mul_f16_w32(arithmetic, ~0U, 0, &pd, &px, &py, nullptr);
  rj_arith_v_add_f16_w32(arithmetic, ~0U, 0, &pr, &pp, &pz, nullptr);
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (uint32_t active = exec_mask; active; active &= active - 1) {
    int lane = __builtin_ctz(active);
    // Source registers have been staged. Compute the raw-half OMOD/CLAMP only for written lanes.
    uint16_t value = rocjitsu::arith::selection_output<16>(uint16_t(result[lane]), mode, flags);
    d[lane] = (d[lane] & ~(0xffffU << shift)) | (uint32_t(value) << shift);
  }
}

// Dispatch both wave widths after one validation, before any partial-wave writes.
template <bool Gfx90a = false, typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  const bool separately_rounded =
      Gfx90a && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if (int error = validate_mad(flags, mode)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  mode = rocjitsu::arith::rdna3_output_modifiers(flags, mode);
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    const auto evaluate = separately_rounded ? separately_rounded_mad : mad;
    evaluate(flags, uint32_t(exec_mask >> offset), uint32_t(mode), d[0] + offset, a[0] + offset,
             b[0] + offset, c[0] + offset);
  }
  return RJ_ARITH_SUCCESS;
}

// MI350X (gfx950) half MAD/MAC: two separately rounded FP16 steps with mandatory
// input/output subnormal flushing at both steps, IEEE-gated sNaN preservation, and
// gated OMOD/CLAMP on the final step.
void separately_rounded_mad_gfx950(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  alignas(64) uint32_t x[32], y[32], z[32], product[32] = {}, result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    x[lane] = input_half(a[lane], mode);
    y[lane] = input_half(b[lane], mode >> 1);
    z[lane] = input_half(c[lane], mode >> 2);
  }
  const uint64_t arithmetic =
      (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
      RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  const uint64_t add_modifiers = mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP);
  const uint32_t *px = x, *py = y, *pz = z, *pp = product;
  uint32_t *pd = product, *pr = result;
  rj_arith_v_mul_f16_w32_gfx950(arithmetic, ~0U, 0, &pd, &px, &py, nullptr);
  rj_arith_v_add_f16_w32_gfx950(arithmetic, ~0U, add_modifiers, &pr, &pp, &pz, nullptr);
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (uint32_t active = exec_mask; active; active &= active - 1) {
    int lane = __builtin_ctz(active);
    d[lane] = (d[lane] & ~(0xffffU << shift)) | ((result[lane] & 0xffffU) << shift);
  }
}

template <typename Mask>
int run_gfx950(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~uint64_t(rocjitsu::arith::kFma16InstructionFlags),
          rocjitsu::arith::kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  mode = rocjitsu::arith::rdna3_output_modifiers(flags, mode);
  const bool exact = rocjitsu::arith::is_exact(flags) || (flags & RJ_ARITH_FP_ROUND_MASK);
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    const auto evaluate = exact ? separately_rounded_mad_gfx950 : mad;
    evaluate(flags, uint32_t(exec_mask >> offset), uint32_t(mode), d[0] + offset, a[0] + offset,
             b[0] + offset, c[0] + offset);
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate half MAD; errors leave both destination storage and FP status unchanged.
int rj_arith_v_mad_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  return run(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate half MAD; errors leave both destination storage and FP status unchanged.
int rj_arith_v_mad_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  return run(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate half MAD; errors leave both destination storage and FP status unchanged.
int rj_arith_v_mad_legacy_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate half MAD; errors leave both destination storage and FP status unchanged.
int rj_arith_v_mad_legacy_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx90a MAD; unsupported semantics leave all outputs unchanged.
int rj_arith_v_mad_f16_w32_gfx90a(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx90a MAD; unsupported semantics leave all outputs unchanged.
int rj_arith_v_mad_f16_w64_gfx90a(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx90a MAD; unsupported semantics leave all outputs unchanged.
int rj_arith_v_mad_legacy_f16_w32_gfx90a(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx90a MAD; unsupported semantics leave all outputs unchanged.
int rj_arith_v_mad_legacy_f16_w64_gfx90a(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_mad_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run_gfx950(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_mad_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *trapsts) {
  return run_gfx950(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_mad_legacy_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return run_gfx950(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_mad_legacy_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return run_gfx950(flags, exec_mask, instruction_flags, d, a, b, c);
}
