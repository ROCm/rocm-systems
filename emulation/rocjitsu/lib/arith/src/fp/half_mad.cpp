// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Legacy half MAD uses a rounded FP32 intermediate before FP16 narrowing.
// Both instruction spellings share this inherited approximation and output controls.

#include "fp/half_mad.h"
#include "common/alu.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/float_output.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

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

// Return a selected half after ABS/NEG, preserving the inherited unflushed input policy.
float input(uint32_t word, uint32_t mode) {
  uint16_t bits = uint16_t(word >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
  return rocjitsu::arith::alu_input(rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(bits)),
                                    mode);
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
    float x = input(a[lane], mode), y = input(b[lane], mode >> 1), z = input(c[lane], mode >> 2);
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

// Dispatch both wave widths after one validation, before any partial-wave writes.
template <typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = validate_mad(flags, mode)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  mode = rocjitsu::arith::rdna3_output_modifiers(flags, mode);
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    mad(flags, uint32_t(exec_mask >> offset), uint32_t(mode), d[0] + offset, a[0] + offset,
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
