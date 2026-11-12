// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Whole-word and selected-halfword moves with EXEC masking.
// Owns public entry points and their validation/dispatch policy.

#include "common/internal.h"
#include "common/simd.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

// Copy the selected source value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_mov_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return rj_arith_v_mov_b32_w32(flags, exec_mask, uint32_t(instruction_flags), d, source);
        });
  }
  const uint32_t known = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (instruction_flags) {
    // ABS precedes NEG; all other bits, including NaN payloads, are preserved.
    uint32_t keep = instruction_flags & RJ_ARITH_ALU_ABS_A ? 0x7fffffffU : UINT32_MAX;
    uint32_t toggle = instruction_flags & RJ_ARITH_ALU_NEG_A ? 0x80000000U : 0;
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[0][lane] = (a[0][lane] & keep) ^ toggle;
      }
    }
    return RJ_ARITH_SUCCESS;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = a[0][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Copy the selected source value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_mov_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return rj_arith_v_mov_b16_w32(flags, exec_mask, uint32_t(instruction_flags), d, source);
        });
  }
  const uint32_t known = RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_NEG_A |
                         RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::move_half_x86_64_v3(exec_mask, instruction_flags, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned sa = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned sd = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t value = (a[0][lane] >> sa) & 65535;
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      value &= 32767;
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      value ^= 32768;
    }

    // MOV modifies the sign bit but does not perform FP arithmetic. Hardware
    // ignores OMOD and CLAMP, including for subnormals and signaling NaNs.
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(65535U << sd)) | (value << sd);
    }
  }
  return RJ_ARITH_SUCCESS;
}

namespace {

// Accepted DPP controls never cross a 16-lane row, so each 32-lane half can
// reuse the existing implementation. Whole-VGPR aliasing cannot couple halves.
template <bool Half>
int move_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a) {
  constexpr auto move = Half ? rj_arith_v_mov_b16_w32 : rj_arith_v_mov_b32_w32;

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = move(flags, 0, mode, nullptr, nullptr)) {
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
    uint32_t *destination = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;

    // Validation above covers both calls: only valid row-enable bits differ.
    int error = move(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                     &destination, &source);
    if (error) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Copy the selected source value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_mov_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move_wave64<false>(flags, exec_mask, instruction_flags, d, a);
}

// Copy the selected source value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_mov_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return move_wave64<true>(flags, exec_mask, instruction_flags, d, a);
}
