// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 FP16/BF16 SWMMAC entry points, with FP32 or packed 16-bit
// results in either wave size. Owns validation and dispatch, plus scalar
// accumulation with the sparse DOT4 stage order and packed-result narrowing.

#include "common/internal.h"
#include "common/packed16.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"
#include "wmma/sparse_layout.h"
#include "wmma/swmmac_float.h"

#include <array>
#include <cmath>
#include <stdint.h>

namespace {

// Truncate FP32 to BF16 precision and return it widened back to FP32,
// retaining sign and exponent bits, including subnormal encodings.
[[gnu::noinline]] float truncate_bf16(float value) {
  // Keep this noinline: GCC 15.2 at -O2 removes the mask when inlined
  // into the packed accumulator's FMA recurrence. See tests/compiler/bf16_truncation.cpp.
  return rocjitsu::arith::as_float(rocjitsu::arith::as_bits(value) & 0xffff0000U);
}

// Accumulate a sparse FP16/BF16 tile into D with resolved semantics.
// Requires valid increasing 2:4 metadata pairs; inputs may overlap D.
// Validation errors leave D unchanged.
template <bool Bf16, int WaveSize, bool Packed>
int swmmac(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
           const uint32_t *const *b, const uint32_t *const *index) {
  const uint32_t known = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B | RJ_ARITH_WMMA_NEG_HI_A |
                         RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C |
                         RJ_ARITH_SWMMAC_INDEX_KEY_1 |
                         (WaveSize == 64 ? RJ_ARITH_SWMMAC_INDEX_KEY_2 : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true,
                                            rocjitsu::arith::kMatrixIgnoredFpFlags)) {
    return error;
  }
  // The shared kernels store 32-lane register sections. Packed Wave64
  // uses two physical VGPRs; FP32 Wave64 uses four.
  uint32_t *dest[Packed ? 4 : 8];
  for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
    constexpr int count = Packed ? 2 : 4;
    dest[reg] = WaveSize == 64 ? d[reg % count] + 32 * (reg / count) : d[reg];
  }
  // Inherited rocjitsu sparse model: four DOT4 steps in compressed K order,
  // narrowing packed accumulators at each step. B negation follows compressed
  // pair position, irrespective of the selected dense element's parity.
  if (rocjitsu::arith::is_exact(flags)) {
    uint32_t result[Packed ? 4 : 8][32] = {};
    unsigned key =
        WaveSize == 64 ? 8 * ((mode >> 7) & 3) : (mode & RJ_ARITH_SWMMAC_INDEX_KEY_1 ? 16 : 0);
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        int reg = row % 8, lane = col + 16 * (row / 8);
        uint32_t acc = Packed ? uint16_t(dest[reg / 2][lane] >> (16 * (reg % 2))) : dest[reg][lane];
        // Borrowed gfx1201 sparse model: apply ABS before NEG to tied C.
        if (mode & RJ_ARITH_WMMA_ABS_C) {
          acc &= Packed ? 0x7fffU : 0x7fffffffU;
        }
        if (mode & RJ_ARITH_WMMA_NEG_C) {
          acc ^= Packed ? 0x8000U : 0x80000000U;
        }
        for (int stage = 0; stage < 4; ++stage) {
          std::array<uint16_t, 4> left, right;
          for (int i = 0; i < 4; ++i) {
            int ck = 4 * stage + i, slot = ck % 2;
            left[i] = uint16_t(rocjitsu::arith::swmmac_a_field<16, WaveSize>(row, ck).read(a));
            if (mode & (slot ? RJ_ARITH_WMMA_NEG_HI_A : RJ_ARITH_WMMA_NEG_LO_A)) {
              left[i] ^= 0x8000;
            }
            int k = rocjitsu::arith::swmmac_selected_k<16, WaveSize>(index, row, ck, key);
            right[i] = uint16_t(rocjitsu::arith::swmmac_b_bits<16, WaveSize>(b, col, k));
            if (mode & (slot ? RJ_ARITH_WMMA_NEG_HI_B : RJ_ARITH_WMMA_NEG_LO_B)) {
              right[i] ^= 0x8000;
            }
          }
          if constexpr (Packed) {
            acc = rocjitsu::arith::packed16::widen<Bf16>(uint16_t(acc));
          }
          acc = rocjitsu::arith::gfx12_dot_bits<Bf16, 4, Packed>(left, right, acc,
                                                                 flags & RJ_ARITH_FP16_OVFL);
        }
        if constexpr (Packed) {
          result[reg / 2][lane] |= acc << (16 * (reg % 2));
        } else {
          result[reg][lane] = acc;
        }
      }
    }
    // Read all input and tied C/D values before any potentially aliasing stores.
    for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        dest[reg][lane] = result[reg][lane];
      }
    }
    return RJ_ARITH_SUCCESS;
  }
  // Decode once for loose scalar/SIMD accumulation; snapshot tied D as C.
  rocjitsu::arith::SwmmacFloatInputs input;
  rocjitsu::arith::swmmac16_prepare<Bf16, WaveSize, Packed>(input, mode, dest, a, b, index);
  bool saturate = (flags & RJ_ARITH_FP16_OVFL) != 0;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::swmmac_float_x86_64_v4<Bf16, Packed>(mode, dest, input, saturate);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::swmmac_float_x86_64_v3<Bf16, Packed>(mode, dest, input, saturate);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if constexpr (Packed) {
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        float acc = input.acc[row][col];
        for (int stage = 0; stage < 4; ++stage) {
          for (int i = 0; i < 4; ++i) {
            int ck = 4 * stage + i;
            float bv = input.b[input.selected[row][ck]][col];
            if (mode & (i % 2 ? RJ_ARITH_WMMA_NEG_HI_B : RJ_ARITH_WMMA_NEG_LO_B)) {
              bv = -bv;
            }
            acc = std::fma(input.a[row][ck], bv, acc);
          }
          // Narrow between stages, not just at the final store.
          if constexpr (Bf16) {
            acc = truncate_bf16(acc);
          } else {
            acc = rocjitsu::arith::as_float(rocjitsu::arith::packed16::widen<false>(
                rocjitsu::arith::float_to_f16(acc, saturate)));
          }
        }
        input.acc[row][col] = acc;
      }
    }
  } else {
    rocjitsu::arith::swmmac_float_accumulate(mode, input);
  }
  uint32_t result[Packed ? 4 : 8][32];
  rocjitsu::arith::swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (int reg = 0; reg < (Packed ? 4 : 8); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      dest[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate gfx12 Wave32 FP16 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<false, 32, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 BF16 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<true, 32, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 FP16 sparse WMMA (K=32) into FP16 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f16_16x16x32_f16_w32(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<false, 32, true>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave32 BF16 sparse WMMA (K=32) into BF16 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<true, 32, true>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 FP16 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<false, 64, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 BF16 sparse WMMA (K=32) into FP32 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<true, 64, false>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 FP16 sparse WMMA (K=32) into FP16 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_f16_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<false, 64, true>(flags, instruction_flags, d, a, b, index);
}

// Evaluate gfx12 Wave64 BF16 sparse WMMA (K=32) into BF16 D.
// D is the input accumulator; index supplies valid 2:4 metadata.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_swmmac_bf16_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
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
  return swmmac<true, 64, true>(flags, instruction_flags, d, a, b, index);
}
