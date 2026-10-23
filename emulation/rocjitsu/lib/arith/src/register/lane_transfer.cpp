// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Transfers between individual VGPR lanes and scalar values.
// Owns public entry points and their validation/dispatch policy.

// Scalar lane-transfer rules follow rocjitsu generated RDNA3/RDNA4 execution
// and vm/amdgpu/register_access.h (lane indices wrap to the wave size).

#include "common/bits.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Validate the instruction's accepted flags before operand access or output writes.
int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rocjitsu::arith::validate(flags, uint32_t(mode), true);
}

} // namespace

// Read the first active VGPR lane into a scalar value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_readfirstlane_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  int lane = exec_mask ? rocjitsu::arith::bit_count_zero<true>(exec_mask) : 0;
  *d = a[0][lane];
  return RJ_ARITH_SUCCESS;
}

// Read the selected VGPR lane into a scalar value.
// Return errors before changing outputs.
int rj_arith_v_readlane_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                const uint32_t *const *a, uint32_t lane) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  *d = a[0][lane & 31];
  return RJ_ARITH_SUCCESS;
}

// Write a scalar value to the selected VGPR lane.
// Return errors before changing outputs.
int rj_arith_v_writelane_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 uint32_t a, uint32_t lane) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  d[0][lane & 31] = a;
  return RJ_ARITH_SUCCESS;
}

// Read the first active VGPR lane into a scalar value for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_readfirstlane_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  int lane = exec_mask ? rocjitsu::arith::bit_count_zero<true>(exec_mask) : 0;
  *d = a[0][lane];
  return RJ_ARITH_SUCCESS;
}

// Read the selected VGPR lane into a scalar value.
// Return errors before changing outputs.
int rj_arith_v_readlane_b32_w64(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                const uint32_t *const *a, uint32_t lane) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  *d = a[0][lane & 63];
  return RJ_ARITH_SUCCESS;
}

// Write a scalar value to the selected VGPR lane.
// Return errors before changing outputs.
int rj_arith_v_writelane_b32_w64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 uint32_t a, uint32_t lane) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  d[0][lane & 63] = a;
  return RJ_ARITH_SUCCESS;
}
