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
#include "wmma/layout.h"

#include <stdint.h>

namespace {

// View gfx12 Wave64 storage as Wave32 register sections, without copying data.
// The delegated kernel must finish source reads before writing destinations.
class Wave32Views {
public:
  const uint32_t *a[2], *b[2], *c[8];
  uint32_t *d[8];

  Wave32Views(uint32_t *const *dst, const uint32_t *const *lhs, const uint32_t *const *rhs,
              const uint32_t *const *acc)
      : a{lhs[0], lhs[0] + 32}, b{rhs[0], rhs[0] + 32} {
    for (int reg = 0; reg < 8; ++reg) {
      c[reg] = acc[reg % 4] + 32 * (reg / 4);
      d[reg] = dst[reg % 4] + 32 * (reg / 4);
    }
  }
};

// Return one unsigned Bits-wide factor at logical (index, k) in gfx12
// Wave32 packing; K must be 16, or 32 for INT4.
template <int Bits, int K = 16> uint32_t element(const uint32_t *const *v, int index, int k) {
  static_assert(K == 16 || (K == 32 && Bits == 4));
  // Eight K elements occupy each lane group; K=32 interleaves these blocks
  // across registers: its K labels swap ISA K bits 3 and 4.
  // CLAMP stages therefore follow physical lane groups, not contiguous ISA K.
  int local = rocjitsu::arith::wmma_gfx12_packed_local(k);
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
int floating(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Captured C ABS/NEG bits are ignored aliases. A/B instruction_flags remain unsupported.
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~uint64_t(RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C),
          rocjitsu::arith::kMatrixIgnoredFpFlags)) {
    return error;
  }
  if (rocjitsu::arith::is_exact(flags)) {
    uint32_t result[8][32];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        int lane = col + 16 * (row / 8), reg = row % 8;
        uint32_t acc = c[reg][lane];
        for (int stage = 0; stage < 2; ++stage) {
          uint8_t left[8], right[8];
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
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::fp8_wmma_x86_64_v3<Bf8A, Bf8B>(d, a, b, c);
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
        acc = __builtin_fmaf(rocjitsu::arith::fp8_to_float<Bf8A>(uint8_t(element<8>(a, row, k))),
                             rocjitsu::arith::fp8_to_float<Bf8B>(uint8_t(element<8>(b, col, k))),
                             acc);
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
int floating_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~uint64_t(RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C),
          rocjitsu::arith::kMatrixIgnoredFpFlags)) {
    return error;
  }
  Wave32Views view(d, a, b, c);
  return floating<Bf8A, Bf8B>(flags, instruction_flags, view.d, view.a, view.b, view.c);
}

// Evaluate a gfx12 Wave32 INT8/INT4 tile with resolved semantics, signedness
// and optional CLAMP. Errors preserve D; D may overlap input registers.
template <int Bits, int K>
int integer(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~uint64_t(RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B |
                                               RJ_ARITH_WMMA_CLAMP))) {
    return error;
  }
#if defined(RJ_ARITH_HAVE_AVX512VNNI)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_ZEN4)) {
    rocjitsu::arith::integer_wmma_avx512vnni<Bits, K>(uint32_t(instruction_flags), d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer_wmma_x86_64_v3<Bits, K>(uint32_t(instruction_flags), d, a, b, c);
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
      const bool clamp = instruction_flags & RJ_ARITH_WMMA_CLAMP;
      const int stages = clamp ? 2 : 1;
      for (int stage = 0; stage < stages; ++stage) {
        for (int local = 0; local < K / stages; ++local) {
          // CLAMP groups K blocks 0+2, then 1+3; one final clamp is not equivalent.
          int k = clamp ? rocjitsu::arith::wmma_gfx12_integer_stage_k(stage, local) : local;
          acc += extend(element<Bits, K>(a, row, k), instruction_flags & RJ_ARITH_WMMA_SIGNED_A) *
                 extend(element<Bits, K>(b, col, k), instruction_flags & RJ_ARITH_WMMA_SIGNED_B);
        }
        if (clamp) {
          acc = rocjitsu::arith::clamp(acc, -INT64_C(2147483648), INT64_C(2147483647));
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
int integer_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(
          flags, instruction_flags & ~uint64_t(RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B |
                                               RJ_ARITH_WMMA_CLAMP))) {
    return error;
  }
  Wave32Views view(d, a, b, c);
  return integer<Bits, K>(flags, instruction_flags, view.d, view.a, view.b, view.c);
}

} // namespace

// Evaluate gfx12 Wave32 FP8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating<false, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 FP8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating<false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 BF8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating<true, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 BF8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
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
  return integer<4, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave32 INT4 WMMA (K=32) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x32_iu4_w32_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return integer<4, 32>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 FP8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating_wave64<false, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 FP8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating_wave64<false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 BF8/FP8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return floating_wave64<true, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 BF8/BF8 WMMA (K=16) into FP32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
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
  return integer_wave64<4, 16>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx12 Wave64 INT4 WMMA (K=32) into INT32 D.
// C supplies the accumulator; D may overlap A, B, or C.
// Errors leave D unchanged; EXEC does not restrict this operation.
int rj_arith_v_wmma_i32_16x16x32_iu4_w64_gfx12(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return integer_wave64<4, 32>(flags, instruction_flags, d, a, b, c);
}
