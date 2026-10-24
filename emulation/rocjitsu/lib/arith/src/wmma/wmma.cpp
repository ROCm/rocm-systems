// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense gfx12 FP16/BF16 WMMA entry points for Wave32 and Wave64, with FP32
// results. Owns validation and scalar execution; exact dot arithmetic lives in dot/.

#include "common/float_formats.h"
#include "common/internal.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"

#include <array>
#include <cmath>
#include <stdint.h>

namespace {

// Evaluate a gfx12 FP16/BF16 tile with already-resolved semantics. D may
// overlap inputs; invalid flags return an error without modifying D.
template <bool Bf16, int WaveSize = 32>
int wmma(uint64_t flags, uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~rocjitsu::arith::kWmmaInstructionFlags, true,
          rocjitsu::arith::kMatrixIgnoredFpFlags)) {
    return error;
  }

  // Each source lane holds eight K elements in Wave32, four in Wave64;
  // output lanes select columns and row groups (rocjitsu shared/mma_exec.h).
  constexpr int OutputRegs = 256 / WaveSize;
  constexpr int KPerLane = 256 / WaveSize;
  constexpr int DestinationRegs = OutputRegs;
  uint32_t result[DestinationRegs][WaveSize] = {};
  // Read one factor, applying the operand's low/high-half sign modifier.
  const auto read_bits = [instruction_flags](const uint32_t *const *v, int index, int k,
                                             int operand) {
    uint16_t value = uint16_t(v[(k % KPerLane) / 2][index + 16 * (k / KPerLane)] >> (16 * (k % 2)));
    if ((instruction_flags >> (operand + 3 * (k % 2))) & 1) {
      value ^= 0x8000;
    }
    return value;
  };
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8) + (WaveSize == 64 ? 32 * ((row / 4) % 2) : 0);
      int reg = row % OutputRegs;
      uint32_t c_bits = c[reg][lane];

      if (instruction_flags & RJ_ARITH_WMMA_ABS_C) {
        c_bits &= 0x7fffffff;
      }
      if (instruction_flags & RJ_ARITH_WMMA_NEG_C) {
        c_bits ^= 0x80000000;
      }
      if (rocjitsu::arith::is_exact(flags)) {
        uint32_t acc = c_bits;
        for (int k = 0; k < 16; k += 4) {
          std::array<uint16_t, 4> left, right;
          for (int j = 0; j < 4; ++j) {
            // GFX12 wave32 processes physical K chunks in order 0,8,4,12;
            // wave64 processes them in order 0,4,8,12.
            int logical = k + j;
            int physical = WaveSize == 32
                               ? (logical & 3) | ((logical & 4) << 1) | ((logical & 8) >> 1)
                               : logical;
            left[j] = read_bits(a, row, physical, 0);
            right[j] = read_bits(b, col, physical, 1);
          }

          acc = rocjitsu::arith::gfx12_dot_bits<Bf16, 4>(left, right, acc,
                                                         flags & RJ_ARITH_FP16_OVFL);
        }

        result[reg][lane] = acc;

      } else {
        float acc = rocjitsu::arith::as_float(c_bits);
        for (int step = 0; step < 16; ++step) {
          int k = step;
          auto left = read_bits(a, row, k, 0), right = read_bits(b, col, k, 1);
          float x =
              Bf16 ? rocjitsu::arith::bf16_to_float(left) : rocjitsu::arith::f16_to_float(left);
          float y =
              Bf16 ? rocjitsu::arith::bf16_to_float(right) : rocjitsu::arith::f16_to_float(right);
          acc = std::fma(x, y, acc);
        }

        result[reg][lane] = rocjitsu::arith::as_bits(acc);
      }
    }
  }

  // Delayed stores are necessary even with exact whole-VGPR aliasing: a
  // destination may overwrite sources consumed by a different output lane.
  for (int reg = 0; reg < DestinationRegs; ++reg) {
    for (int lane = 0; lane < WaveSize; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate gfx12 Wave32 FP16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return wmma<false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 BF16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return wmma<true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 FP16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 BF16 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}
