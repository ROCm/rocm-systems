// SPDX-License-Identifier: MIT

// Dense gfx11 WMMA entry points for FP16/BF16 and INT8/INT4, in both wave
// sizes. Implements replicated input layouts and interleaved output rows,
// including gfx11 accumulation stages, with scalar execution and SIMD dispatch.

#include "common/float_formats.h"
#include "common/internal.h"
#include "common/packed16.h"
#include "dot/gfx11_dot2.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

// Evaluate a gfx11 FP16/BF16 tile with already-resolved semantics. Packed
// results preserve the unselected half of D; errors leave all of D unchanged.
template <bool Bf16, int Lanes, bool Packed = false>
int wmma(uint64_t flags, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  const uint64_t known = 63 | (Packed ? RJ_ARITH_WMMA_HIGH_C_D : 0);
  if (mode & ~known) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error =
          rocjitsu::arith::validate(flags, 0, true, rocjitsu::arith::matrix_ignored_fp_flags)) {
    return error;
  }
  // gfx11 layouts replicate A/B in each 16-lane group and interleave output
  // rows across groups (rocjitsu shared/mma_exec.h).
  constexpr int Groups = Lanes / 16, Registers = 256 / Lanes;
  const unsigned shift = (mode & RJ_ARITH_WMMA_HIGH_C_D) ? 16 : 0;
  uint32_t result[Registers][Lanes];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int group = row % Groups, lane = group * 16 + col, reg = row / Groups;
      uint32_t acc = c[reg][lane];
      if constexpr (Packed) {
        acc = (acc >> shift) & 65535;
      }
      if (mode & RJ_ARITH_WMMA_ABS_C) {
        acc &= Packed ? 0x7fff : 0x7fffffff;
      }
      if (mode & RJ_ARITH_WMMA_NEG_C) {
        acc ^= Packed ? 0x8000 : 0x80000000;
      }
      float loose = rocjitsu::arith::as_float(
          Packed ? rocjitsu::arith::packed16::widen<Bf16>(uint16_t(acc)) : acc);
      for (int k = 0; k < 16; k += 2) {
        uint32_t av = a[k / 2][group * 16 + row];
        uint32_t bv = b[k / 2][group * 16 + col];
        av ^= ((mode & RJ_ARITH_WMMA_NEG_LO_A) ? 0x8000U : 0) |
              ((mode & RJ_ARITH_WMMA_NEG_HI_A) ? 0x80000000U : 0);
        bv ^= ((mode & RJ_ARITH_WMMA_NEG_LO_B) ? 0x8000U : 0) |
              ((mode & RJ_ARITH_WMMA_NEG_HI_B) ? 0x80000000U : 0);
        if (rocjitsu::arith::is_exact(flags)) {
          if constexpr (Packed) {
            acc = rocjitsu::arith::gfx11_dot2_packed16<Bf16>(av, bv, av >> 16, bv >> 16, acc,
                                                             flags & RJ_ARITH_FP16_OVFL);
          } else {
            acc = rocjitsu::arith::gfx11_dot2_f32<Bf16>(av, bv, av >> 16, bv >> 16, acc);
          }
        } else {
          for (int half = 0; half < 2; ++half) {
            uint16_t x = uint16_t(av >> (16 * half)), y = uint16_t(bv >> (16 * half));
            loose = std::fma(
                Bf16 ? rocjitsu::arith::bf16_to_float(x) : rocjitsu::arith::f16_to_float(x),
                Bf16 ? rocjitsu::arith::bf16_to_float(y) : rocjitsu::arith::f16_to_float(y), loose);
          }
          if constexpr (Packed) {
            // RDNA3 narrows after every two products, RDNA4 after four.
            acc = Bf16 ? rocjitsu::arith::as_bits(loose) >> 16
                       : rocjitsu::arith::float_to_f16(loose, flags & RJ_ARITH_FP16_OVFL);
            loose =
                rocjitsu::arith::as_float(rocjitsu::arith::packed16::widen<Bf16>(uint16_t(acc)));
          }
        }
      }
      if constexpr (Packed) {
        result[reg][lane] = (d[reg][lane] & ~(65535U << shift)) | (acc << shift);
      } else {
        result[reg][lane] =
            rocjitsu::arith::is_exact(flags) ? acc : rocjitsu::arith::as_bits(loose);
      }
    }
  }
  // Finish all source and preserved-half reads before any aliasing D writes.
  for (int reg = 0; reg < Registers; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate gfx11 Wave32 FP16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_f16_w32(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return wmma<false, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave32 BF16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return wmma<true, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave64 FP16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx11 Wave64 BF16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}
