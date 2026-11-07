// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 single/pair scaled widening from FP4/FP8/BF8. Shares scale decoding,
// destination packing and validation across formats; preserves the inherited
// Rocjitsu model without claiming newly established hardware bit-exactness.

#include "conversion/scaled_widen.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "conversion/scaled_widen_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <stdint.h>

namespace {
using namespace rocjitsu::arith;
using Source = ScaledSource;
using Destination = ScaledDestination;

// Decode one OCP narrow value, preserving its sign including on NaNs.
template <Source Format> float decode(uint32_t code) {
  if constexpr (Format == Source::Fp4) {
    constexpr float magnitude[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return as_float(as_bits(magnitude[code & 7]) | ((code & 8) << 28));
  } else {
    return fp8_to_float<Format == Source::Bf8>(uint8_t(code));
  }
}

// Widen the active lanes, retaining unselected destination bits. Validate all
// controls before touching operands, including for an empty EXEC mask.
template <int Lanes, Source Src, Destination Dst, bool Packed>
int convert(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t selectors =
      Packed && Src != Source::Fp4 ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3;
  constexpr uint64_t modifiers =
      selectors | (!Packed && Dst == Destination::F16 ? RJ_ARITH_ALU_HIGH_D : 0);
  if (int error = validate(flags, mode & ~modifiers)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  int shift = Packed && Src != Source::Fp4 ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)
                                           : 8 * int((mode >> 16) & 3);
  [[maybe_unused]] const int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    scaled_widen_x86_64_v4(Lanes, exec_mask, Src, Dst, Packed, flags & RJ_ARITH_FP16_OVFL, shift,
                           destination_shift, d, a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    scaled_widen_x86_64_v3(Lanes, exec_mask, Src, Dst, Packed, flags & RJ_ARITH_FP16_OVFL, shift,
                           destination_shift, d, a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr int words = Packed && Dst == Destination::F32 ? 2 : 1;
  uint32_t result[words][Lanes];
  // Stage the entire result before stores so either input may overlap either
  // destination word, including when both destination pointers are identical.
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint32_t raw = a[0][lane] >> shift;
    uint32_t exponent = (b[0][lane] >> 23) & 255;
    // Exponent zero denotes 2^-127, not FP32 zero. Multiplication by the
    // exact power of two has the inherited double-then-FP32 model's one rounding.
    float scale = as_float(exponent ? exponent << 23 : 0x00400000U);
    uint32_t values[Packed ? 2 : 1];
    for (int element = 0; element < (Packed ? 2 : 1); ++element) {
      if (exponent == 255) {
        values[element] = Dst == Destination::F32   ? 0x7fc00000U
                          : Dst == Destination::F16 ? 0x7e00U
                                                    : 0x7fc0U;
      } else {
        values[element] = encode_scaled_widen<Dst>(
            decode<Src>(raw >> (element * (Src == Source::Fp4 ? 4 : 8))) * scale,
            flags & RJ_ARITH_FP16_OVFL);
      }
    }
    result[0][lane] = values[0];
    if constexpr (Packed) {
      if constexpr (Dst == Destination::F32) {
        result[1][lane] = values[1];
      } else {
        result[0][lane] |= values[1] << 16;
      }
    } else if constexpr (Dst == Destination::F16) {
      result[0][lane] =
          (d[0][lane] & ~(0xffffU << destination_shift)) | (values[0] << destination_shift);
    }
  }
  for (int word = 0; word < words; ++word) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[word][lane] = result[word][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Scale and widen FP8 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Fp8, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Fp8, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to F32, one element per active lane.
int rj_arith_v_cvt_scalef32_f32_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<32, Source::Fp8, Destination::F32, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP8 to F32, one element per active lane.
int rj_arith_v_cvt_scalef32_f32_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<64, Source::Fp8, Destination::F32, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Bf8, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen BF8 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Bf8, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen BF8 to F32, one element per active lane.
int rj_arith_v_cvt_scalef32_f32_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<32, Source::Bf8, Destination::F32, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to F32, one element per active lane.
int rj_arith_v_cvt_scalef32_f32_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<64, Source::Bf8, Destination::F32, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP4 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_fp4_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Fp4, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP4 to F32, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f32_fp4_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Fp4, Destination::F32, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Fp8, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Fp8, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to F16, one element per active lane.
int rj_arith_v_cvt_scalef32_f16_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<32, Source::Fp8, Destination::F16, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP8 to F16, one element per active lane.
int rj_arith_v_cvt_scalef32_f16_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<64, Source::Fp8, Destination::F16, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Bf8, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen BF8 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Bf8, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen BF8 to F16, one element per active lane.
int rj_arith_v_cvt_scalef32_f16_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<32, Source::Bf8, Destination::F16, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to F16, one element per active lane.
int rj_arith_v_cvt_scalef32_f16_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                               uint64_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<64, Source::Bf8, Destination::F16, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP4 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_fp4_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::Fp4, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP4 to F16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_f16_fp4_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::Fp4, Destination::F16, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

// Scale and widen FP8 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_fp8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Fp8, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP8 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_fp8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Fp8, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_bf8_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Bf8, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen BF8 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_bf8_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Bf8, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP4 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_fp4_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Fp4, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}

// Scale and widen FP4 to BF16, two elements per active lane.
int rj_arith_v_cvt_scalef32_pk_bf16_fp4_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Fp4, Destination::Bf16, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b);
}
