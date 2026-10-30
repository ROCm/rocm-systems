// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 INT8/INT4 SWMMAC entry points for Wave32 and Wave64. Validates
// flags and dispatches decoded operands to scalar or SIMD accumulation,
// preserving the instruction's stage-by-stage saturation or wraparound.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"
#include "wmma/swmmac_integer_impl.h"

#include <stdint.h>

namespace {

// Accumulate a sparse INT8/INT4 tile into D with resolved semantics.
// Requires valid increasing 2:4 metadata pairs; inputs may overlap D.
// Validation errors leave D unchanged.
template <int Bits, int K, int WaveSize = 32>
int swmmac(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
           const uint32_t *const *b, const uint32_t *const *index) {
  const uint32_t known = RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP |
                         rocjitsu::arith::SwmmacIndex<Bits, K, WaveSize>::mask;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  // Expose eight 32-lane destination sections to both wave-size backends.
  uint32_t *dest[8];
  for (int reg = 0; reg < 8; ++reg) {
    dest[reg] = WaveSize == 64 ? d[reg % 4] + 32 * (reg / 4) : d[reg];
  }
  rocjitsu::arith::SwmmacIntegerInputs input;
  rocjitsu::arith::swmmac_integer_prepare<Bits, K, WaveSize>(input, mode, dest, a, b, index);
  bool clamp = mode & RJ_ARITH_WMMA_CLAMP;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::swmmac_integer_x86_64_v4(K, clamp, dest, input);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::swmmac_integer_x86_64_v3(K, clamp, dest, input);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int64_t acc = input.acc[row][col];
      // Use wide sums, but clamp at each hardware stage, not only at the end.
      for (int stage = 0; stage < 2; ++stage) {
        for (int local = 0; local < K / 4; ++local) {
          int ck = rocjitsu::arith::swmmac_integer_ck(stage, local);
          acc += int64_t(input.a[row][ck]) * input.b[input.selected[row][ck]][col];
        }
        if (clamp) {
          acc = rocjitsu::arith::clamp(acc, -INT64_C(2147483648), INT64_C(2147483647));
        }
      }
      result[row % 8][col + 16 * (row / 8)] = uint32_t(acc);
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      dest[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate gfx12 Wave32 INT8 sparse WMMA (K=32) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x32_iu8_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<8, 32>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 INT4 sparse WMMA (K=32) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x32_iu4_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<4, 32>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 INT4 sparse WMMA (K=64) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x64_iu4_w32(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<4, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 INT8 sparse WMMA (K=32) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x32_iu8_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<8, 32, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 INT4 sparse WMMA (K=32) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x32_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<4, 32, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 INT4 sparse WMMA (K=64) into INT32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_i32_16x16x64_iu4_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<4, 64, 64>(flags, instruction_flags, d, a, b, index);
}
