// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Per-lane conditional selection between two source registers.
// Owns public entry points and their validation/dispatch policy.

// Source sign-bit modifiers follow rocjitsu's conditional-selection helpers.

#include "register/cndmask.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Select between two sources by lane condition for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Half>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t condition) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return run<Half>(flags, exec_mask, uint32_t(mode), d, source, b, condition);
        });
  }
  const uint32_t known =
      RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
      RJ_ARITH_ALU_NEG_B |
      (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::cndmask_x86_64_v4<Half>(exec_mask, mode, d[0], a[0], b[0], condition);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  // Half-word selection is faster in the portable path than AVX2's
  // shift/merge/masked-word stores, including with nondefault selectors.
  if constexpr (!Half) {
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::cndmask_x86_64_v3(exec_mask, mode, d[0], a[0], b[0], condition);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif

  const uint32_t sign = Half ? 0x8000 : 0x80000000;
  unsigned sa = Half && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
  unsigned sb = Half && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
  unsigned sd = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
  uint32_t clear_a = mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX;
  uint32_t clear_b = mode & RJ_ARITH_ALU_ABS_B ? ~sign : UINT32_MAX;
  uint32_t flip_a = mode & RJ_ARITH_ALU_NEG_A ? sign : 0;
  uint32_t flip_b = mode & RJ_ARITH_ALU_NEG_B ? sign : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = ((a[0][lane] >> sa) & clear_a) ^ flip_a;
    uint32_t y = ((b[0][lane] >> sb) & clear_b) ^ flip_b;
    uint32_t value = ((condition >> lane) & 1) ? y : x;
    if constexpr (Half) {
      value = (d[0][lane] & ~(65535u << sd)) | ((value & 65535) << sd);
    }
    result[lane] = value;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Select between two sources by lane condition for Wave64. Reject invalid controls before
// changing outputs.
template <bool Half>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, uint64_t condition) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Half>(flags, 0, mode, nullptr, nullptr, nullptr, 0)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }

  // DPP stays within 16-lane rows, so whole-VGPR aliases are safe across halves.
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32;
    const uint32_t *source_b = b[0] + half * 32;
    if (int error = run<Half>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                              &dest, &source_a, &source_b, uint32_t(condition >> (half * 32)))) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Select between A and B using the per-lane condition for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_cndmask_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t condition) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, condition);
}

// Select between A and B using the per-lane condition for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_cndmask_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t condition) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, condition);
}

// Select between A and B using the per-lane condition for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_cndmask_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint64_t condition) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, condition);
}

// Select between A and B using the per-lane condition for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_cndmask_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint64_t condition) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, condition);
}
