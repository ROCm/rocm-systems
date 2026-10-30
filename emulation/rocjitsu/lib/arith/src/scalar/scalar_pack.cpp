// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar word/byte packing and packed source selection.
// Owns public entry points and their validation/dispatch policy.

// Semantics follow rocjitsu generated/shared/execute_shared.h. Bit spreading and
// compression implement the same per-bit operations without scalar loops.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Validate the instruction's accepted flags before operand access or output writes.
int validate(uint64_t flags, uint32_t mode) {
  return rocjitsu::arith::validate(
      flags, mode, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

// Compress or expand active bits at quad granularity, according to Compress.
template <typename T, bool Compress> T quad(T a) {
  a |= a >> 1;
  a |= a >> 2;
  a &= T(0x1111111111111111ULL);
  if constexpr (!Compress) {
    return a * 15;
  }
  a = (a | (a >> 3)) & T(0x0303030303030303ULL);
  a = (a | (a >> 6)) & T(0x000f000f000f000fULL);
  a = (a | (a >> 12)) & T(0x000000ff000000ffULL);
  if constexpr (sizeof(T) == 8) {
    a = (a | (a >> 24)) & 65535;
  }
  return a;
}

} // namespace

// Pack the selected source halfwords.
// Return errors before changing outputs.
int rj_arith_s_pack_ll_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ((a >> 0) & 65535) | ((b >> 0) << 16);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Pack the selected source halfwords.
// Return errors before changing outputs.
int rj_arith_s_pack_lh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ((a >> 0) & 65535) | ((b >> 16) << 16);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Pack the selected source halfwords.
// Return errors before changing outputs.
int rj_arith_s_pack_hl_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ((a >> 16) & 65535) | ((b >> 0) << 16);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Pack the selected source halfwords.
// Return errors before changing outputs.
int rj_arith_s_pack_hh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = ((a >> 16) & 65535) | ((b >> 16) << 16);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Execute BITREPLICATE_B64_B32 on scalar values.
// Return errors before changing outputs.
int rj_arith_s_bitreplicate_b64_b32(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint32_t a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t x = a;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  uint64_t result = x | (x << 1);
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Select a scalar source using SCC.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cselect_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = (input_scc & 1) ? a : b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Select a scalar source using SCC.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_cselect_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t b, uint32_t input_scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = (input_scc & 1) ? a : b;
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Reduce active bits to a quad mask.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_quadmask_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = quad<uint32_t, true>(a);
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Reduce active bits to a quad mask.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_quadmask_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = quad<uint64_t, true>(a);
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Expand active bits to whole quads.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_wqm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t result = quad<uint32_t, false>(a);
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}

// Expand active bits to whole quads.
// Return errors before changing outputs. SCC is an ordinary instruction result.
int rj_arith_s_wqm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t *scc) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint64_t result = quad<uint64_t, false>(a);
  *d = result;
  uint32_t cc = result != 0;
  __builtin_memcpy(scc, &cc, sizeof(cc));
  return RJ_ARITH_SUCCESS;
}
