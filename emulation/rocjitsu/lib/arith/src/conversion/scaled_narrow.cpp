// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 scaled FP4/FP8/BF8 narrowing. Shares inherited OCP rounding with unscaled
// conversions while retaining SCALEF32 scale and partial-destination rules.

#include "conversion/scaled_narrow.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "conversion/scaled_narrow_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <stdint.h>

namespace {
using namespace rocjitsu::arith;
using Source = ScaledNarrowSource;
using Destination = ScaledNarrowDestination;

// Decode one input element before scaling. Single stochastic conversions read
// only the low half of FP16/BF16 sources.
template <Source Format> float decode(uint32_t raw, bool flush_input) {
  if constexpr (Format == Source::F16) {
    return f16_to_float(uint16_t(raw));
  } else if constexpr (Format == Source::Bf16) {
    return bf16_to_float(uint16_t(raw));
  } else {
    return as_float(flush_input && (raw & 0x7f800000U) == 0 ? raw & 0x80000000U : raw);
  }
}

// Scale then narrow active lanes. Scale exponent 255 preserves the destination;
// every other exponent divides by 2^(exponent-127). Validate before any access.
template <int Lanes, Source Src, Destination Dst, bool Stochastic>
int convert(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr bool fp4 = Dst == Destination::Fp4;
  constexpr uint64_t known = Stochastic || fp4 ? RJ_ARITH_CVT_BYTE_3 : RJ_ARITH_ALU_HIGH_D;
  if (int error = validate(flags, mode & ~known,
                           Src == Source::F32 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // Packed FP32 uses A/B data and C scale; packed FP16/BF16 uses both A halves
  // and B scale. Stochastic forms always use A data, B seed and C scale.
  const uint32_t *scale = Stochastic || Src == Source::F32 ? c[0] : b[0];
  int shift = Stochastic || fp4 ? 8 * int((mode >> 16) & 3) : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const uint32_t *upper_a = nullptr;
  if constexpr (fp4 && Stochastic && Src == Source::F32) {
    upper_a = a[1];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    scaled_narrow_x86_64_v4<Dst, Stochastic>(Lanes, exec_mask, Src, flags & RJ_ARITH_FP16_OVFL,
                                             flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, shift, d[0],
                                             a[0], b[0], scale, upper_a);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    scaled_narrow_x86_64_v3<Dst, Stochastic>(Lanes, exec_mask, Src, flags & RJ_ARITH_FP16_OVFL,
                                             flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, shift, d[0],
                                             a[0], b[0], scale, upper_a);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t selected = (Stochastic || fp4 ? 0xffU : 0xffffU) << shift;
  const bool flush_input = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    result[lane] = d[0][lane];
    uint32_t exponent = (scale[lane] >> 23) & 255;
    if (exponent == 255) {
      continue;
    }
    // The reciprocal power of two is exactly representable, including 2^-127
    // at exponent 254. This retains the old double-division model's FP32 rounding.
    float inverse_scale = as_float(exponent == 254 ? 0x00400000U : (254 - exponent) << 23);
    uint32_t seed = Stochastic ? b[0][lane] : 0;
    uint32_t value = scaled_narrow_encode<Dst, Stochastic>(
        as_bits(decode<Src>(a[0][lane], flush_input) * inverse_scale), seed,
        flags & RJ_ARITH_FP16_OVFL);
    if constexpr (!Stochastic || fp4) {
      uint32_t upper;
      if constexpr (Src == Source::F32) {
        if constexpr (Stochastic) {
          upper = a[1][lane];
        } else {
          upper = b[0][lane];
        }
      } else {
        upper = a[0][lane] >> 16;
      }
      uint32_t next_seed = Stochastic ? narrow_next_seed(seed) : 0;
      value |= scaled_narrow_encode<Dst, Stochastic>(
                   as_bits(decode<Src>(upper, flush_input) * inverse_scale), next_seed,
                   flags & RJ_ARITH_FP16_OVFL)
               << (fp4 ? 4 : 8);
    }
    result[lane] = (result[lane] & ~selected) | (value << shift);
  }
  // Delay writes until all source and scale values are consumed, permitting
  // aliases with any operand and retaining inactive lanes.
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Scale F32 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale F32 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale F32 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F32 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F32 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale F32 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale F32 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F32 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F16 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale F16 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::F16, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale F16 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F16 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F16, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F16 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale F16 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::F16, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale F16 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale F16 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F16, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale BF16 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale BF16 and narrow to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp8_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Bf16, Destination::Fp8, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale BF16 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale BF16 and narrow to FP8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_fp8_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return convert<64, Source::Bf16, Destination::Fp8, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale BF16 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale BF16 and narrow to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_bf8_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Bf16, Destination::Bf8, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale BF16 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale BF16 and narrow to BF8 with seeded stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_bf8_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return convert<64, Source::Bf16, Destination::Bf8, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and pack two F32 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and pack two F32 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and pack two F32 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and pack two F32 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and pack two F16 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and pack two F16 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<64, Source::F16, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and pack two F16 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and pack two F16 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b,
                                                     const uint32_t *const *c) {
  return convert<64, Source::F16, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and pack two BF16 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and pack two BF16 inputs into E2M1 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk_fp4_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<64, Source::Bf16, Destination::Fp4, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and pack two BF16 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and pack two BF16 inputs into E2M1 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk_fp4_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<64, Source::Bf16, Destination::Fp4, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}
