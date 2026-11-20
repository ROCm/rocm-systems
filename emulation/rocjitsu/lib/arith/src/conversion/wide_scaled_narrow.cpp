// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide scaled narrowing: gfx950 packs 32 FP6/BF6 values; gfx1250 packs 8/16
// FP4/FP8/BF8/FP6/BF6 values, sharing format rounding and register staging. Shares finite
// rounding with FP4 while retaining source interleaving and per-element seeds.

#include "common/float_encoding.h"
#include "common/validation.h"
#include "conversion/scaled_narrow.h"
#include "conversion/scaled_narrow_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <stdint.h>

namespace {
using namespace rocjitsu::arith;
using Source = ScaledNarrowSource;
using Destination = ScaledNarrowDestination;

// Decode a FP32 value or the low FP16/BF16 half without modifying FP controls.
template <Source Src> float decode(uint32_t raw, bool flush_input) {
  if constexpr (Src == Source::F16) {
    return f16_to_float(uint16_t(raw));
  } else if constexpr (Src == Source::Bf16) {
    return bf16_to_float(uint16_t(raw));
  } else {
    return as_float(flush_input && (raw & 0x7f800000U) == 0 ? raw & 0x80000000U : raw);
  }
}

// Scale and pack each active lane. gfx950 FP32 RNE interleaves two 16-register
// inputs; stochastic FP32 uses one 32-register input. FP16/BF16 occupy 16 A
// registers. gfx1250 consumes one contiguous source and divides by full FP32
// scale for 8/16 elements. Staging results before stores permits source/destination overlap.
template <int Lanes, Source Src, Destination Dst, bool Stochastic, bool FullScale = false>
int convert(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error =
          validate(flags, mode, Src == Source::F32 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  constexpr int bits = Dst == Destination::Fp4                                ? 4
                       : (Dst == Destination::Fp6 || Dst == Destination::Bf6) ? 6
                                                                              : 8;
  constexpr int elements = FullScale ? (bits == 6 ? 16 : 8) : 32;
  constexpr int words = elements * bits / 32;
  const uint32_t *scale = Stochastic || (!FullScale && Src == Source::F32) ? c[0] : b[0];
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    wide_scaled_narrow_x86_64_v4<Dst, Stochastic, FullScale>(
        Lanes, exec_mask, Src, d, a, b, scale, flags & RJ_ARITH_FP16_OVFL,
        flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    wide_scaled_narrow_x86_64_v3<Dst, Stochastic, FullScale>(
        Lanes, exec_mask, Src, d, a, b, scale, flags & RJ_ARITH_FP16_OVFL,
        flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    return RJ_ARITH_SUCCESS;
  }
#endif
  const bool flush_input = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  uint32_t result[words][Lanes];
  uint64_t write_mask = exec_mask;
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint32_t exponent = (scale[lane] >> 23) & 255;
    if (!FullScale && exponent == 255) {
      write_mask &= ~(1ULL << lane);
      continue;
    }
    float inverse_scale = as_float(exponent == 254 ? 0x00400000U : (254 - exponent) << 23);
    uint32_t seed = Stochastic ? b[0][lane] : 0;
    for (int word = 0; word < words; ++word) {
      result[word][lane] = 0;
    }
    for (int element = 0; element < elements; ++element) {
      uint32_t raw;
      if constexpr (Src == Source::F32) {
        if constexpr (Stochastic || FullScale) {
          raw = a[element][lane];
        } else {
          // Index each array before selecting: GCC 13.3 miscompiles indexing
          // a conditional array base here, including under UBSan.
          raw = element % 2 ? b[element / 2][lane] : a[element / 2][lane];
        }
      } else {
        raw = a[element / 2][lane] >> ((element % 2) * 16);
      }
      float scaled;
      if constexpr (FullScale) {
        scaled = decode<Src>(raw, flush_input) / as_float(scale[lane]);
      } else {
        scaled = decode<Src>(raw, flush_input) * inverse_scale;
      }
      uint32_t value =
          scaled_narrow_encode<Dst, Stochastic>(as_bits(scaled), seed, flags & RJ_ARITH_FP16_OVFL);
      int bit = element * bits, word = bit / 32, shift = bit % 32;
      result[word][lane] |= value << shift;
      if (shift > 32 - bits) {
        result[word + 1][lane] |= value >> (32 - shift);
      }
      if constexpr (Stochastic) {
        seed = narrow_next_seed(seed);
      }
    }
  }
  for (int word = 0; word < words; ++word) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((write_mask >> lane) & 1) {
        d[word][lane] = result[word][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Scale and narrow 32 F32 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_2xpk16_fp6_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 F32 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_2xpk16_fp6_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 F32 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F32 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_fp6_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and narrow 32 F16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_fp6_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, Source::F16, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and narrow 32 F16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<64, Source::F16, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 BF16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_fp6_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and narrow 32 BF16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_fp6_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<64, Source::Bf16, Destination::Fp6, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and narrow 32 BF16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_bf16_w32_gfx950(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 BF16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_fp6_bf16_w64_gfx950(
    uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<64, Source::Bf16, Destination::Fp6, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 F32 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_2xpk16_bf6_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 F32 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_2xpk16_bf6_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 F32 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F32 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<64, Source::F32, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_bf6_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and narrow 32 F16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_bf6_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, Source::F16, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, nullptr);
}

// Scale and narrow 32 F16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 F16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<64, Source::F16, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

// Scale and narrow 32 BF16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_bf6_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and narrow 32 BF16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk32_bf6_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<64, Source::Bf16, Destination::Bf6, false>(flags, exec_mask, instruction_flags, d,
                                                            a, b, nullptr);
}

// Scale and narrow 32 BF16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_bf16_w32_gfx950(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Scale and narrow 32 BF16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk32_bf6_bf16_w64_gfx950(
    uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<64, Source::Bf16, Destination::Bf6, true>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

// Divide and narrow 16 F32 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_fp6_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::F32, Destination::Fp6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 F32 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_fp6_f32_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp6, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow 16 F16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_fp6_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 F16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_fp6_f16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp6, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow 16 BF16 values to FP6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_fp6_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 BF16 values to FP6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_fp6_bf16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp6, true, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}

// Divide and narrow 16 F32 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_bf6_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::F32, Destination::Bf6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 F32 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_bf6_f32_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf6, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow 16 F16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_bf6_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Bf6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 F16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_bf6_f16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Bf6, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow 16 BF16 values to BF6 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk16_bf6_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                      uint64_t instruction_flags,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Bf6, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow 16 BF16 values to BF6 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk16_bf6_bf16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Bf6, true, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}

// Divide and narrow eight F32 values to FP4 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp4_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F32, Destination::Fp4, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F32 values to FP4 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp4_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp4, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight F16 values to FP4 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp4_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp4, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F16 values to FP4 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp4_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp4, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight BF16 values to FP4 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp4_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp4, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight BF16 values to FP4 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp4_bf16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp4, true, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}

// Divide and narrow eight F32 values to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F32, Destination::Fp8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F32 values to FP8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Fp8, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight F16 values to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp8_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Fp8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F16 values to FP8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp8_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Fp8, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight BF16 values to FP8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_fp8_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Fp8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight BF16 values to FP8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_fp8_bf16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Fp8, true, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}

// Divide and narrow eight F32 values to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_bf8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F32, Destination::Bf8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F32 values to BF8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_bf8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F32, Destination::Bf8, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight F16 values to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_bf8_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, Source::F16, Destination::Bf8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight F16 values to BF8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_bf8_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                       uint64_t instruction_flags,
                                                       uint32_t *const *d, const uint32_t *const *a,
                                                       const uint32_t *const *b,
                                                       const uint32_t *const *c) {
  return convert<32, Source::F16, Destination::Bf8, true, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, c);
}

// Divide and narrow eight BF16 values to BF8 with nearest-even rounding.
int rj_arith_v_cvt_scalef32_pk8_bf8_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, Source::Bf16, Destination::Bf8, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Divide and narrow eight BF16 values to BF8 with stochastic rounding.
int rj_arith_v_cvt_scalef32_sr_pk8_bf8_bf16_w32_gfx1250(
    uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return convert<32, Source::Bf16, Destination::Bf8, true, true>(flags, exec_mask,
                                                                 instruction_flags, d, a, b, c);
}
