// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DPP8/DPP16 source routing, encoding validation, and effective EXEC masks.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Writes all 32 permuted source words; out must not overlap a. Inactive sources
// become positive zero unless FI is set. Does not access destination operands.
void dpp8_source(uint32_t exec_mask, uint64_t instruction_flags, uint32_t *out, const uint32_t *a);

// Checks descriptor fields and the RDNA4 control encoding; low 32 bits are
// validated by the instruction. Zero denotes no DPP modifier.
bool valid_dpp(uint64_t instruction_flags);

// Writes all permuted source words and returns the destination EXEC mask after
// row/bank/source filtering. out must not overlap a. Valid descriptor required.
uint32_t dpp16_source(uint32_t exec_mask, uint64_t instruction_flags, uint32_t *out,
                      const uint32_t *a);

// Restrict a Wave64 DPP16 row mask to one 32-lane half-wave (half is 0 or 1).
inline uint64_t dpp_half_wave_instruction_flags(uint64_t instruction_flags, int half) {
  if (!(instruction_flags & RJ_ARITH_DPP16)) {
    return instruction_flags;
  }
  uint64_t rows = (instruction_flags & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
  return (instruction_flags & ~RJ_ARITH_DPP_ROW_MASK) |
         (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
}

// Applies DPP to one 32-bit source VGPR, then executes vector-only arithmetic.
// execute(exec_mask, source_a) must validate flags before touching operands, leave all
// state unchanged for zero EXEC, and support a temporary source pointer array.
// The zero-EXEC call validates the instruction's own flags before source access.
template <typename Execute>
int execute_dpp(uint32_t exec_mask, uint64_t instruction_flags, const uint32_t *const *a,
                Execute execute) {
  if (!(instruction_flags & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !valid_dpp(instruction_flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }

  // Validate the underlying operation before routing can read operands or change outputs.
  int error = execute(0, a);
  if (error || !exec_mask) {
    return error;
  }
  uint32_t permuted[32];
  const uint32_t *source = permuted;
  if (instruction_flags & RJ_ARITH_DPP8) {
    dpp8_source(exec_mask, instruction_flags, permuted, a[0]);
  } else {
    exec_mask = dpp16_source(exec_mask, instruction_flags, permuted, a[0]);
  }
  return execute(exec_mask, &source);
}

} // namespace rocjitsu::arith
