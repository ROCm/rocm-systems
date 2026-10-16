// SPDX-License-Identifier: MIT

// DPP8/DPP16 source routing, encoding validation, and effective EXEC masks.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Writes all 32 permuted source words; out must not overlap a. Inactive sources
// become positive zero unless FI is set. Does not access destination operands.
void dpp8_source(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *out,
                 const uint32_t *a);

// Evaluate dpp8 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dpp8_x86_64_v3(uint32_t exec_mask, uint32_t selectors, bool fi, uint32_t *out,
                    const uint32_t *a);

// Evaluate dpp8 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dpp8_x86_64_v4(uint32_t exec_mask, uint32_t selectors, bool fi, uint32_t *out,
                    const uint32_t *a);

// Checks descriptor fields and the RDNA4 control encoding; low 32 bits are
// validated by the instruction. Zero denotes no DPP modifier.
bool valid_dpp(uint64_t mode);

// Writes all permuted source words and returns the destination EXEC mask after
// row/bank/source filtering. out must not overlap a. Valid descriptor required.
uint32_t dpp16_source(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *out,
                      const uint32_t *a);

// Evaluate dpp16 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t dpp16_x86_64_v3(uint32_t exec_mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a);

// Evaluate dpp16 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t dpp16_x86_64_v4(uint32_t exec_mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a);

// Applies DPP to one 32-bit source VGPR, then executes vector-only arithmetic.
// execute(exec_mask, source_a) must validate flags before touching operands, leave all
// state unchanged for zero EXEC, and support a temporary source pointer array.
// The zero-EXEC call validates the instruction's own flags before source access.
template <typename Execute>
int execute_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode, const uint32_t *const *a,
                Execute execute) {
  if (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !valid_dpp(mode)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }

  // Validate the underlying operation before routing can read operands or change outputs.
  int error = execute(0, a);
  if (error || !exec_mask) {
    return error;
  }
  uint32_t permuted[32];
  const uint32_t *source = permuted;
  if (mode & RJ_ARITH_DPP8) {
    dpp8_source(flags, exec_mask, mode, permuted, a[0]);
  } else {
    exec_mask = dpp16_source(flags, exec_mask, mode, permuted, a[0]);
  }
  return execute(exec_mask, &source);
}

} // namespace rocjitsu::arith
