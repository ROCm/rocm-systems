// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stage gfx1250 accumulators and pack adjacent output rows in VGPR lane order.
#pragma once
#include "common/compiler.h"
#if defined(RJ_ARITH_TARGET_X86_64_V4)
#include "common/x86_64/half_x86_64_v4.h"
#else
#include "common/x86_64/half_x86_64_v3.h"
#endif
#include "wmma/gfx1250_half.h"
#include <cstring>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
using Gfx1250PackedWords [[gnu::vector_size(64)]] = uint32_t;
using Gfx1250PackedFloat = __m512;
#else
using Gfx1250PackedWords [[gnu::vector_size(32)]] = uint32_t;
using Gfx1250PackedFloat = __m256;
#endif

template <Wmma1250Form Form>
inline void gfx1250_widen_packed_c_pair(Gfx1250PackedWords bits, Gfx1250PackedWords keep,
                                        Gfx1250PackedWords negate, Gfx1250PackedWords &even_out,
                                        Gfx1250PackedWords &odd_out) {
  using Words = Gfx1250PackedWords;
  if constexpr (Form == Wmma1250Form::F16) {
#if defined(RJ_ARITH_TARGET_X86_64_V4)
    const auto raw = bit_cast<__m512i>(bits);
    const __m256i lo_h = _mm512_cvtepi32_epi16(raw);
    const __m256i hi_h = _mm512_cvtepi32_epi16(_mm512_srli_epi32(raw, 16));
    Words lo = bit_cast<Words>(_mm512_cvtph_ps(lo_h));
    Words hi = bit_cast<Words>(_mm512_cvtph_ps(hi_h));
#else
    const auto raw = bit_cast<__m256i>(bits);
    const __m256i order = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0,
                                           1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    const __m256i split = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(raw, order), 0xd8);
    Words lo = bit_cast<Words>(_mm256_cvtph_ps(_mm256_castsi256_si128(split)));
    Words hi = bit_cast<Words>(_mm256_cvtph_ps(_mm256_extracti128_si256(split, 1)));
#endif
    // The C encoding retains the signaling bit until accumulation.
    Words h_lo = bits & 0xffffu;
    Words h_hi = bits >> 16;
    lo &= ~(bit_cast<Words>((h_lo & 0x7e00u) == 0x7c00u) & 0x00400000u);
    hi &= ~(bit_cast<Words>((h_hi & 0x7e00u) == 0x7c00u) & 0x00400000u);
    even_out = (lo & keep) ^ negate;
    odd_out = (hi & keep) ^ negate;
  } else {
    even_out = ((bits << 16) & keep) ^ negate;
    odd_out = ((bits & 0xffff0000u) & keep) ^ negate;
  }
}

template <Wmma1250Form Form>
inline void gfx1250_load_sparse_accumulators(const uint32_t *const *c, uint64_t controls,
                                             float tile[16][16]) {
  using Words = Gfx1250PackedWords;
  constexpr int width = sizeof(Words) / sizeof(uint32_t);
  constexpr bool packed = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  if constexpr (packed) {
    if (controls & RJ_ARITH_WMMA_INLINE_C) {
      const auto value = gfx1250_accumulator(c[0][0], controls);
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          tile[row][col] = value;
        }
      }
      return;
    }
  }
  const uint32_t keep = (controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fffffffu : 0xffffffffu;
  const uint32_t negate = (controls & RJ_ARITH_WMMA_NEG_C) ? 0x80000000u : 0;
  for (int row = 0; row < 16; ++row) {
    const int reg = (row % 8) / (packed ? 2 : 1);
    const int group = row / 8;
    for (int col = 0; col < 16; col += width) {
      Words bits;
      std::memcpy(&bits, c[reg] + 16 * group + col, sizeof(bits));
      if constexpr (packed) {
        const int shift = (row % 2) * 16;
        Words h = (bits >> shift) & 0xffffu;
        if constexpr (Form == Wmma1250Form::F16) {
#ifdef RJ_ARITH_TARGET_X86_64_V4
          auto result = half_input(bit_cast<__m512i>(bits), shift, 0);
#else
          auto result = half_input(bit_cast<__m256i>(bits), shift, 0);
#endif
          bits = bit_cast<Words>(result);
          // The C encoding retains the signaling bit until accumulation.
          bits &= ~(bit_cast<Words>((h & 0x7e00u) == 0x7c00u) & 0x00400000u);
        } else {
          bits = h << 16;
        }
      }
      bits = (bits & keep) ^ negate;
      std::memcpy(tile[row] + col, &bits, sizeof(bits));
    }
  }
}

template <Wmma1250Form Form>
inline void gfx1250_load_accumulators(const uint32_t *const *c, uint64_t controls,
                                      float tile[16][16]) {
  using Words = Gfx1250PackedWords;
  constexpr int width = sizeof(Words) / sizeof(uint32_t);
  constexpr bool packed = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  if constexpr (packed) {
    if (controls & RJ_ARITH_WMMA_INLINE_C) {
      const auto value = gfx1250_accumulator(c[0][0], controls);
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          tile[row][col] = value;
        }
      }
      return;
    }
  }
  const uint32_t keep_scalar = (controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fffffffu : 0xffffffffu;
  const uint32_t negate_scalar = (controls & RJ_ARITH_WMMA_NEG_C) ? 0x80000000u : 0;
  const Words keep = Words{} + keep_scalar;
  const Words negate = Words{} + negate_scalar;
  if constexpr (packed) {
    for (int reg = 0; reg < 4; ++reg) {
      for (int group = 0; group < 2; ++group) {
        const int row = 2 * reg + 8 * group;
        for (int col = 0; col < 16; col += width) {
          Words bits, lo, hi;
          std::memcpy(&bits, c[reg] + 16 * group + col, sizeof(bits));
          gfx1250_widen_packed_c_pair<Form>(bits, keep, negate, lo, hi);
          std::memcpy(tile[row] + col, &lo, sizeof(lo));
          std::memcpy(tile[row + 1] + col, &hi, sizeof(hi));
        }
      }
    }
  } else {
    for (int reg = 0; reg < 8; ++reg) {
      for (int group = 0; group < 2; ++group) {
        const int row = reg + 8 * group;
        for (int col = 0; col < 16; col += width) {
          Words bits;
          std::memcpy(&bits, c[reg] + 16 * group + col, sizeof(bits));
          bits = (bits & keep) ^ negate;
          std::memcpy(tile[row] + col, &bits, sizeof(bits));
        }
      }
    }
  }
}

template <Wmma1250Form Form, bool Sparse = false>
inline Gfx1250PackedWords gfx1250_narrow_result(Gfx1250PackedFloat values, bool compatibility,
                                                bool saturate) {
  using Words = Gfx1250PackedWords;
  Words bits = bit_cast<Words>(values);
  Words upper = bits >> 16;
  if constexpr (Form == Wmma1250Form::F16) {
    Words value = bit_cast<Words>(half_narrow(values, false));
    // F16C quiets signaling inputs. The WMMA narrowing rule retains their
    // top payload bits and keeps at least one fraction bit set.
    Words signaling =
        bit_cast<Words>(((bits & 0x7fffffffu) > 0x7f800000u) & ((bits & 0x00400000u) == 0u));
    value &= ~(signaling & 0x0200u);
    value |= signaling & bit_cast<Words>((value & 0x03ffu) == 0u) & 1u;
    if (saturate) {
      value = ((value & 0x7fffu) == 0x7c00u) ? value - 1u : value;
    }
    return value;
  } else {
    if (compatibility) {
      return upper;
    }
    Words rounded = (bits + 0x7fffu + (upper & 1u)) >> 16;
    Words value;
    if constexpr (Sparse) {
      // Sparse BF16 retains a marker if narrowing discards any NaN payload bit.
      Words special = upper | (bit_cast<Words>((bits & 0xffffu) != 0u) & 1u);
      value = ((bits & 0x7f800000u) == 0x7f800000u) ? special : rounded;
    } else {
      value = ((bits & 0x7fffffffu) > 0x7f800000u) ? (upper | 0x40u) : rounded;
    }
    if (saturate) {
      value = ((value & 0x7fffu) == 0x7f80u) ? value - 1u : value;
    }
    return value;
  }
}

template <Wmma1250Form Form, bool Sparse = false>
inline void gfx1250_store_results(uint64_t flags, uint32_t *const *d, const float result[16][16],
                                  bool saturate_f16 = false) {
  if constexpr (Form == Wmma1250Form::F32F16 || Form == Wmma1250Form::F32Bf16) {
    if constexpr (Sparse) {
      gfx1250_write_result<Form>(flags, d, result, saturate_f16);
    } else {
      for (int reg = 0; reg < 8; ++reg) {
        std::memcpy(d[reg], result[reg], sizeof(result[reg]));
        std::memcpy(d[reg] + 16, result[reg + 8], sizeof(result[reg + 8]));
      }
    }
  } else {
    using Words = Gfx1250PackedWords;
    constexpr int width = sizeof(Words) / sizeof(uint32_t);
    const bool compatibility =
        !Sparse && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
    const bool saturate = (Form == Wmma1250Form::F16 && saturate_f16) ||
                          (!compatibility && (flags & RJ_ARITH_FP16_OVFL));
    for (int reg = 0; reg < 4; ++reg) {
      for (int group = 0; group < 2; ++group) {
        int row = 2 * reg + 8 * group;
        for (int col = 0; col < 16; col += width) {
          Gfx1250PackedFloat lo, hi;
          std::memcpy(&lo, result[row] + col, sizeof(lo));
          std::memcpy(&hi, result[row + 1] + col, sizeof(hi));
          auto words = gfx1250_narrow_result<Form, Sparse>(lo, compatibility, saturate) |
                       (gfx1250_narrow_result<Form, Sparse>(hi, compatibility, saturate) << 16);
          std::memcpy(d[reg] + 16 * group + col, &words, sizeof(words));
        }
      }
    }
  }
}

#if defined(RJ_ARITH_TARGET_X86_64_V4)
// Load C directly into 16 register-resident ZMM accumulators (one per output row).
template <Wmma1250Form Form>
[[gnu::always_inline]] inline void gfx1250_load_acc_zmm(const uint32_t *const *c, uint64_t controls,
                                                        __m512 (&acc)[16]) {
  using Words = Gfx1250PackedWords;
  constexpr bool packed = Form == Wmma1250Form::F16 || Form == Wmma1250Form::Bf16;
  if constexpr (packed) {
    if (controls & RJ_ARITH_WMMA_INLINE_C) {
      const __m512 value = _mm512_set1_ps(gfx1250_accumulator(c[0][0], controls));
      RJ_ARITH_UNROLL(16)
      for (int row = 0; row < 16; ++row) {
        acc[row] = value;
      }
      return;
    }
  }
  const uint32_t keep_scalar = (controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fffffffu : 0xffffffffu;
  const uint32_t negate_scalar = (controls & RJ_ARITH_WMMA_NEG_C) ? 0x80000000u : 0;
  const Words keep = Words{} + keep_scalar;
  const Words negate = Words{} + negate_scalar;
  if constexpr (packed) {
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < 4; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        const int row = 2 * reg + 8 * group;
        Words bits = bit_cast<Words>(_mm512_loadu_si512(c[reg] + 16 * group));
        Words lo, hi;
        gfx1250_widen_packed_c_pair<Form>(bits, keep, negate, lo, hi);
        acc[row] = bit_cast<__m512>(lo);
        acc[row + 1] = bit_cast<__m512>(hi);
      }
    }
  } else {
    RJ_ARITH_UNROLL(8)
    for (int reg = 0; reg < 8; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        Words bits = bit_cast<Words>(_mm512_loadu_si512(c[reg] + 16 * group));
        acc[reg + 8 * group] = bit_cast<__m512>((bits & keep) ^ negate);
      }
    }
  }
}

// Store 16 register-resident ZMM accumulators directly to D in ascending VGPR order.
template <Wmma1250Form Form, bool Sparse = false>
[[gnu::always_inline]] inline void gfx1250_store_acc_zmm(uint64_t flags, uint32_t *const *d,
                                                         const __m512 (&acc)[16],
                                                         bool saturate_f16 = false) {
  if constexpr (Form == Wmma1250Form::F32F16 || Form == Wmma1250Form::F32Bf16) {
    RJ_ARITH_UNROLL(8)
    for (int reg = 0; reg < 8; ++reg) {
      _mm512_storeu_ps(d[reg], acc[reg]);
      _mm512_storeu_ps(d[reg] + 16, acc[reg + 8]);
    }
  } else {
    const bool compatibility =
        !Sparse && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
    const bool saturate = (Form == Wmma1250Form::F16 && saturate_f16) ||
                          (!compatibility && (flags & RJ_ARITH_FP16_OVFL));
    RJ_ARITH_UNROLL(4)
    for (int reg = 0; reg < 4; ++reg) {
      RJ_ARITH_UNROLL(2)
      for (int group = 0; group < 2; ++group) {
        const int row = 2 * reg + 8 * group;
        Gfx1250PackedWords words =
            gfx1250_narrow_result<Form, Sparse>(acc[row], compatibility, saturate) |
            (gfx1250_narrow_result<Form, Sparse>(acc[row + 1], compatibility, saturate) << 16);
        _mm512_storeu_si512(d[reg] + 16 * group, bit_cast<__m512i>(words));
      }
    }
  }
}

// Vectorized AVX-512 ordered FMA preserving Rocjitsu NaN priority (A, then B, then accumulator).
inline __m512 gfx1250_matrix_nan_fma_v4(__m512 a, __m512 b, __m512 c) {
  const __m512 res = _mm512_fmadd_ps(a, b, c);
  const __mmask16 res_nan = _mm512_cmp_ps_mask(res, res, _CMP_UNORD_Q);
  if (!res_nan) {
    return res;
  }
  const __m512i quiet = _mm512_set1_epi32(0x00400000);
  const __mmask16 c_nan = _mm512_cmp_ps_mask(c, c, _CMP_UNORD_Q);
  const __mmask16 b_nan = _mm512_cmp_ps_mask(b, b, _CMP_UNORD_Q);
  const __mmask16 a_nan = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
  __m512i fix = _mm512_set1_epi32(int(0xffc00000U));
  fix = _mm512_mask_or_epi32(fix, c_nan, _mm512_castps_si512(c), quiet);
  fix = _mm512_mask_or_epi32(fix, b_nan, _mm512_castps_si512(b), quiet);
  fix = _mm512_mask_or_epi32(fix, a_nan, _mm512_castps_si512(a), quiet);
  return _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_castps_si512(res), res_nan, fix));
}
#endif
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
