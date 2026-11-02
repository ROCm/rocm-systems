// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 FP8/BF8 SWMMAC entry points with FP32 results in either wave
// size. Decodes sparse operands, adapts Wave64 destinations, and dispatches
// loose accumulation or evaluates the bit-exact two-stage DOT8 model.

#include "common/float_formats.h"
#include "common/internal.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_float_impl.h"

#include <stdint.h>

namespace {

// Accumulate a sparse FP8/BF8 tile into D with resolved semantics.
// Requires valid increasing 2:4 metadata pairs; inputs may overlap D.
// Validation errors leave D unchanged.
template <bool Bf8A, bool Bf8B, int WaveSize = 32>
int swmmac(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
           const uint32_t *const *b, const uint32_t *const *index) {
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(rocjitsu::arith::SwmmacIndex<8, 32, WaveSize>::mask),
          rocjitsu::arith::kMatrixIgnoredFpFlags)) {
    return error;
  }
  // Existing accumulation kernels use eight 32-lane destination sections.
  // Wave64 stores the upper four sections in the upper lanes of D[0..3].
  uint32_t *dest[8];
  for (int reg = 0; reg < 8; ++reg) {
    dest[reg] = WaveSize == 64 ? d[reg % 4] + 32 * (reg / 4) : d[reg];
  }
  int key = rocjitsu::arith::SwmmacIndex<8, 32, WaveSize>::shift(mode);
  // Compressed K remains contiguous for sparse DOT8 steps. Snapshot all
  // results before any stores because D can overlap A, B or the index VGPR.
  if (rocjitsu::arith::is_exact(flags)) {
    uint32_t result[8][32];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        int reg = row % 8, out_lane = col + 16 * (row / 8);
        uint32_t acc = dest[reg][out_lane];
        for (int stage = 0; stage < 2; ++stage) {
          uint8_t left[8], right[8];
          for (int i = 0; i < 8; ++i) {
            int ck = stage * 8 + i;
            left[i] = uint8_t(rocjitsu::arith::swmmac_a_field<8, WaveSize>(row, ck).read(a));
            int k = rocjitsu::arith::swmmac_selected_k<8, WaveSize>(index, row, ck, key);
            right[i] = uint8_t(rocjitsu::arith::swmmac_b_bits<8, WaveSize>(b, col, k));
          }
          acc = rocjitsu::arith::gfx12_dot8_f32<Bf8A, Bf8B>(left, right, acc);
        }
        result[reg][out_lane] = acc;
      }
    }
    for (int reg = 0; reg < 8; ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        dest[reg][lane] = result[reg][lane];
      }
    }
    return RJ_ARITH_SUCCESS;
  }
  // Loose backends share decoded FP32 operands, independent of FP8/BF8 packing.
  rocjitsu::arith::SwmmacFloatInputs input;
  for (int row = 0; row < 16; ++row) {
    for (int ck = 0; ck < 16; ++ck) {
      input.a[row][ck] = rocjitsu::arith::fp8_to_float<Bf8A>(
          uint8_t(rocjitsu::arith::swmmac_a_field<8, WaveSize>(row, ck).read(a)));
      input.selected[row][ck] =
          uint8_t(rocjitsu::arith::swmmac_selected_k<8, WaveSize>(index, row, ck, key));
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int col = 0; col < 16; ++col) {
      input.b[k][col] = rocjitsu::arith::fp8_to_float<Bf8B>(
          uint8_t(rocjitsu::arith::swmmac_b_bits<8, WaveSize>(b, col, k)));
    }
  }
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      input.acc[row][col] = rocjitsu::arith::as_float(dest[row % 8][col + 16 * (row / 8)]);
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::swmmac_float_x86_64_v4<false, false>(0, dest, input, false);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::swmmac_float_x86_64_v3<false, false>(0, dest, input, false);
    return RJ_ARITH_SUCCESS;
  }
#endif
  rocjitsu::arith::swmmac_float_accumulate(0, input);
  uint32_t result[8][32];
  rocjitsu::arith::swmmac_float_pack<false, false>(result, input.acc, false);
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      dest[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate gfx12 Wave32 FP8/FP8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<false, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 FP8/BF8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<false, true>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 BF8/FP8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<true, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 BF8/BF8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<true, true>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 FP8/FP8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<false, false, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 FP8/BF8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<false, true, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 BF8/FP8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<true, false, 64>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 BF8/BF8 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return swmmac<true, true, 64>(flags, instruction_flags, d, a, b, index);
}
