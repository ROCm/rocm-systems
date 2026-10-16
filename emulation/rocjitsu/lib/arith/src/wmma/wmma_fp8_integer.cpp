// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense gfx12 FP8/BF8 and INT8/INT4 WMMA entry points. Handles packed operand
// layouts, Wave32/Wave64 adaptation, validation, and scalar/SIMD dispatch;
// floating-point bit-exact accumulation uses the shared DOT8 model.

#include "common/float_formats.h"
#include "common/internal.h"
#include "common/simd.h"
#include "dot/dot.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdint.h>

namespace {

// Return one unsigned Bits-wide factor at logical (index, k) in gfx12
// Wave32 packing; K must be 16, or 32 for INT4.
template <int Bits, int K = 16> uint32_t element(const uint32_t *const *v, int index, int k) {
  static_assert(K == 16 || (K == 32 && Bits == 4));
  // Eight K elements occupy each lane group; K=32 interleaves these blocks
  // across registers (rocjitsu shared/mma_exec.h).
  int local = k % 8 + 8 * (k / 16);
  int lane = index + 16 * ((k / 8) % 2);
  return (v[local * Bits / 32][lane] >> ((local * Bits) % 32)) & ((1u << Bits) - 1);
}

// Commit a staged FP32/INT32 tile to eight 32-lane destination sections.
// Ascending stores make the last destination register win if D entries alias.
void store(uint32_t *const *d, const uint32_t (&result)[8][32]) {
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      d[reg][lane] = result[reg][lane];
    }
  }
}

// Evaluate a gfx12 Wave32 FP8/BF8 tile with resolved semantics. D may alias
// inputs; unsupported modifiers return an error without modifying D.
template <bool Bf8A, bool Bf8B>
int floating(uint64_t flags, uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  // Captured C ABS/NEG bits are ignored aliases. A/B modifiers remain unsupported.
  if (int error =
          rocjitsu::arith::validate(flags, modifiers & ~(RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C),
                                    true, rocjitsu::arith::matrix_ignored_fp_flags)) {
    return error;
  }
  if (rocjitsu::arith::is_exact(flags)) {
    uint32_t result[8][32];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        int lane = col + 16 * (row / 8), reg = row % 8;
        uint32_t acc = c[reg][lane];
        for (int stage = 0; stage < 2; ++stage) {
          std::array<uint8_t, 8> left, right;
          for (int i = 0; i < 8; ++i) {
            left[i] = uint8_t(element<8>(a, row, stage * 8 + i));
            right[i] = uint8_t(element<8>(b, col, stage * 8 + i));
          }
          acc = rocjitsu::arith::gfx12_dot8_f32<Bf8A, Bf8B>(left, right, acc);
        }
        result[reg][lane] = acc;
      }
    }
    store(d, result);
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::fp8_wmma_x86_64_v3(Bf8A, Bf8B, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      uint32_t bits = c[reg][lane];
      float acc = rocjitsu::arith::as_float(bits);
      for (int k = 0; k < 16; ++k) {
        acc = std::fma(rocjitsu::arith::fp8_to_float<Bf8A>(uint8_t(element<8>(a, row, k))),
                       rocjitsu::arith::fp8_to_float<Bf8B>(uint8_t(element<8>(b, col, k))), acc);
      }
      result[reg][lane] = rocjitsu::arith::as_bits(acc);
    }
  }
  store(d, result);
  return RJ_ARITH_SUCCESS;
}

// Evaluate a gfx12 Wave64 FP8/BF8 tile with resolved semantics, preserving
// the Wave32 evaluator's aliasing and error guarantees.
template <bool Bf8A, bool Bf8B>
int floating_wave64(uint64_t flags, uint32_t modifiers, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error =
          rocjitsu::arith::validate(flags, modifiers & ~(RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C),
                                    true, rocjitsu::arith::matrix_ignored_fp_flags)) {
    return error;
  }
  // RDNA4 7.12.2: view upper Wave64 lanes as extra 32-lane registers.
  // The delegated kernel reads all sources before writing these views.
  const uint32_t *av[] = {a[0], a[0] + 32}, *bv[] = {b[0], b[0] + 32};
  const uint32_t *cv[8];
  uint32_t *dv[8];
  for (int reg = 0; reg < 8; ++reg) {
    cv[reg] = c[reg % 4] + 32 * (reg / 4);
    dv[reg] = d[reg % 4] + 32 * (reg / 4);
  }
  return floating<Bf8A, Bf8B>(flags, modifiers, dv, av, bv, cv);
}

// Evaluate a gfx12 Wave32 INT8/INT4 tile with resolved semantics, signedness
// and optional CLAMP. Errors preserve D; D may overlap input registers.
template <int Bits, int K>
int integer(uint64_t flags, uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
            const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags,
          modifiers & ~(RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP),
          true)) {
    return error;
  }
#if defined(RJ_ARITH_HAVE_AVX512VNNI)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_ZEN4) {
    rocjitsu::arith::integer_wmma_avx512vnni(Bits, K, modifiers, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::integer_wmma_x86_64_v3(Bits, K, modifiers, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  // Interpret an extracted Bits-wide value with the requested signedness.
  const auto extend = [](uint32_t bits, bool is_signed) -> int64_t {
    return int64_t(bits) - (is_signed && (bits & (1u << (Bits - 1))) ? (1u << Bits) : 0);
  };
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      uint32_t bits = c[reg][lane];
      int64_t acc = int64_t(bits) - ((bits >> 31) ? (INT64_C(1) << 32) : 0);
      const bool clamp = modifiers & RJ_ARITH_WMMA_CLAMP;
      const int stages = clamp ? 2 : 1;
      for (int stage = 0; stage < stages; ++stage) {
        for (int local = 0; local < K / stages; ++local) {
          // CLAMP groups K blocks 0+2, then 1+3; one final clamp is not equivalent.
          int k = clamp ? 16 * (local / 8) + 8 * stage + local % 8 : local;
          acc += extend(element<Bits, K>(a, row, k), modifiers & RJ_ARITH_WMMA_SIGNED_A) *
                 extend(element<Bits, K>(b, col, k), modifiers & RJ_ARITH_WMMA_SIGNED_B);
        }
        if (clamp) {
          acc = std::clamp(acc, -INT64_C(2147483648), INT64_C(2147483647));
        }
      }
      // Unsigned conversion implements modulo 2^32 without signed overflow.
      result[reg][lane] = uint32_t(acc);
    }
  }
  store(d, result);
  return RJ_ARITH_SUCCESS;
}

// Evaluate a gfx12 Wave64 integer tile with resolved semantics. CLAMP keeps
// the two captured saturation stages; D may alias inputs and errors preserve D.
template <int Bits, int K>
int integer_wave64(uint64_t flags, uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags,
          modifiers & ~(RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B | RJ_ARITH_WMMA_CLAMP),
          true)) {
    return error;
  }
  // RDNA4 7.12.2: view upper Wave64 lanes as extra 32-lane registers.
  // The delegated kernel reads all sources before writing these views.
  const uint32_t *av[] = {a[0], a[0] + 32}, *bv[] = {b[0], b[0] + 32};
  const uint32_t *cv[8];
  uint32_t *dv[8];
  for (int reg = 0; reg < 8; ++reg) {
    cv[reg] = c[reg % 4] + 32 * (reg / 4);
    dv[reg] = d[reg % 4] + 32 * (reg / 4);
  }
  return integer<Bits, K>(flags, modifiers, dv, av, bv, cv);
}

} // namespace

// Evaluate gfx12 Wave32 FP8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
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
  return floating<false, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 FP8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
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
  return floating<false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 BF8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32(uint64_t flags, uint64_t instruction_flags,
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
  return floating<true, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 BF8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32(uint64_t flags, uint64_t instruction_flags,
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
  return floating<true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 INT8 WMMA (K=16) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x16_iu8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
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
  return integer<8, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 INT4 WMMA (K=16) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x16_iu4_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
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
  return integer<4, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 INT4 WMMA (K=32) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x32_iu4_w32(uint64_t flags, uint64_t instruction_flags,
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
  return integer<4, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 FP8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
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
  return floating_wave64<false, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 FP8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
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
  return floating_wave64<false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 BF8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64(uint64_t flags, uint64_t instruction_flags,
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
  return floating_wave64<true, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 BF8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64(uint64_t flags, uint64_t instruction_flags,
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
  return floating_wave64<true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 INT8 WMMA (K=16) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
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
  return integer_wave64<8, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 INT4 WMMA (K=16) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x16_iu4_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
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
  return integer_wave64<4, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 INT4 WMMA (K=32) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x32_iu4_w64(uint64_t flags, uint64_t instruction_flags,
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
  return integer_wave64<4, 32>(flags, instruction_flags, d, a, b, c);
}
