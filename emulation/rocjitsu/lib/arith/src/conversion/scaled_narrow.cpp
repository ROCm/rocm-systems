// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx950 scaled FP4/FP8/BF8 narrowing. Shares inherited OCP rounding with unscaled
// conversions while retaining SCALEF32 scale and partial-destination rules.

#include "conversion/scaled_narrow.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "common/wave_alignment.h"
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
    uint16_t h = uint16_t(raw);
    return f16_to_float(flush_input && (h & 0x7c00U) == 0 ? uint16_t(h & 0x8000U) : h);
  } else if constexpr (Format == Source::Bf16) {
    uint32_t b = uint32_t(uint16_t(raw)) << 16;
    return as_float(flush_input && (b & 0x7f800000U) == 0 ? (b & 0x80000000U) : b);
  } else {
    return as_float(flush_input && (raw & 0x7f800000U) == 0 ? (raw & 0x80000000U) : raw);
  }
}

// Scale then narrow active lanes. Scale exponent 255 emits NaN (FP8/BF8) or
// signed max finite (FP4); every other exponent divides by 2^(exponent-127).
template <int Lanes, Source Src, Destination Dst, bool Stochastic>
int convert(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  constexpr bool fp4 = Dst == Destination::Fp4;
  constexpr uint64_t known = Stochastic || fp4 ? RJ_ARITH_CVT_BYTE_3 : RJ_ARITH_ALU_HIGH_D;
  if (int error = validate(flags, mode & ~known, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // Packed FP32 uses A/B data and C scale; packed FP16/BF16 uses both A halves
  // and B scale. Stochastic forms always use A data, B seed and C scale.
  const uint32_t *scale = Stochastic || Src == Source::F32 ? c[0] : b[0];
  int shift = Stochastic || fp4 ? 8 * int((mode >> 16) & 3) : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  [[maybe_unused]] const uint32_t *upper_a = nullptr;
  if constexpr (fp4 && Stochastic && Src == Source::F32) {
    upper_a = a[1];
  }
  [[maybe_unused]] const auto level =
      wave_cpu_level(flags, !Stochastic || Src != Source::Bf16 || Dst != Destination::Fp8 ||
                                (exec_mask & (exec_mask - 1)));
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (level >= RJ_ARITH_CPU_X86_64_V4) {
    scaled_narrow_x86_64_v4<Dst, Stochastic>(Lanes, exec_mask, Src, flags & RJ_ARITH_FP16_OVFL,
                                             flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, shift, d[0],
                                             a[0], b[0], scale, upper_a);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (level >= RJ_ARITH_CPU_X86_64_V3) {
    scaled_narrow_x86_64_v3<Dst, Stochastic>(Lanes, exec_mask, Src, flags & RJ_ARITH_FP16_OVFL,
                                             flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, shift, d[0],
                                             a[0], b[0], scale, upper_a);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr uint32_t sig_mask = scaled_sig_mask(Src);
  uint32_t selected = (Stochastic || fp4 ? 0xffU : 0xffffU) << shift;
  const bool flush_input = (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) != 0;
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    result[lane] = d[0][lane];
    uint32_t exponent = (scale[lane] >> 23) & 255;
    // The reciprocal power of two is exactly representable, including 2^-127
    // at exponent 254. Exponent 255 substitutes a signed NaN into the encoder.
    float inverse_scale = as_float(exponent == 255   ? 0x3f800000U
                                   : exponent == 254 ? 0x00400000U
                                                     : (254 - exponent) << 23);
    auto scale_elem = [&](uint32_t raw_elem) {
      float decoded = decode<Src>(raw_elem, flush_input);
      if (exponent == 255) {
        return (as_bits(decoded) & 0x80000000U) | 0x7fc00000U;
      }
      return as_bits(decoded * inverse_scale);
    };
    uint32_t seed = Stochastic ? b[0][lane] : 0;
    uint32_t value = scaled_narrow_encode<Dst, Stochastic, true>(
        scale_elem(a[0][lane]), seed, flags & RJ_ARITH_FP16_OVFL, sig_mask);
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
      value |= scaled_narrow_encode<Dst, Stochastic, true>(scale_elem(upper), next_seed,
                                                           flags & RJ_ARITH_FP16_OVFL, sig_mask)
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
