// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 (x86-64-v4, without VNNI) operations and dispatch for the shared
// dense INT8/INT4 WMMA algorithm. Uses 512-bit VPMADDWD for byte pairs and
// VPMADDUBSW + VPMADDWD for packed nibble quartets.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/integer_wmma.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Each 128-bit VPSHUFB lane holds four packed words.
__m512i word_shuffle(int indices) {
  return _mm512_add_epi8(_mm512_set1_epi32(indices),
                         _mm512_set4_epi32(0x0c0c0c0c, 0x08080808, 0x04040404, 0));
}

// Pack four nibbles from one hardware stage into bytes for VPMADDUBSW.
template <bool Adjacent> __m512i packed_nibbles(__m512i words, int k) {
  if constexpr (!Adjacent) {
    return _mm512_and_si512(k & 4 ? _mm512_srli_epi16(words, 4) : words, _mm512_set1_epi8(15));
  }
  __m512i bytes = _mm512_shuffle_epi8(words, word_shuffle(k & 4 ? 0x03030202 : 0x01010000));
  __m512i nibbles =
      _mm512_mask_blend_epi8(UINT64_C(0xaaaaaaaaaaaaaaaa), bytes, _mm512_srli_epi16(bytes, 4));
  return _mm512_and_si512(nibbles, _mm512_set1_epi8(15));
}

// Gfx11 clamped IU8 consumes adjacent byte pairs. Otherwise interleave the
// two pairs of each word as (0,2) and (1,3) to widen in place.
template <bool Adjacent> __m512i byte_pair(__m512i words, int k, __m512i mask) {
  if constexpr (Adjacent) {
    words = _mm512_shuffle_epi8(words, word_shuffle(k & 2 ? 0x03030202 : 0x01010000));
  } else if (!(k & 2)) {
    words = _mm512_slli_epi16(words, 8);
  }
  return _mm512_and_si512(_mm512_srai_epi16(words, 8), mask);
}

// Word-pair dots cover the full IU8 range. IU4 quartet dots accumulate exact
// halfword partials, including the signed-A column correction, until stage end.
template <int Bits, bool Rdna3, bool Clamp, bool Positive = false> class Products {
public:
  using V = __m512i;
  static constexpr int width = 16;
  static constexpr int products = Bits == 4 ? 4 : 2;
  static constexpr bool biased = Bits == 4;
  static constexpr bool staged_clamp = Clamp && !Positive;
  static constexpr bool early_store = true;
  static constexpr bool narrow_clamp = Bits == 4 && staged_clamp;
  struct Accumulator {
    V value, base;
  };

  explicit Products(uint32_t flags) {
    bool signed_a = flags & RJ_ARITH_WMMA_SIGNED_A;
    bool signed_b = flags & RJ_ARITH_WMMA_SIGNED_B;
    bias_a = _mm512_set1_epi8(signed_a ? 8 : 0);
    bias_b = _mm512_set1_epi8(signed_b ? 8 : 0);
    mask_a = _mm512_set1_epi16(signed_a ? -1 : 255);
    mask_b = _mm512_set1_epi16(signed_b ? -1 : 255);
  }

  static V load(const uint32_t *p) { return _mm512_loadu_si512(p); }
  static void store(uint32_t *p, V v) { _mm512_storeu_si512(p, v); }
  static V broadcast(uint32_t word) { return _mm512_set1_epi32(word); }

  V unpack_a(V word, int k) const {
    if constexpr (Bits == 4) {
      return _mm512_xor_si512(packed_nibbles < Rdna3 && staged_clamp > (word, k), bias_a);
    } else {
      return byte_pair < Rdna3 && staged_clamp > (word, k, mask_a);
    }
  }

  V unpack_b(V word, int k) const {
    if constexpr (Bits == 4) {
      __m512i bytes = packed_nibbles < Rdna3 && staged_clamp > (word, k);
      return _mm512_sub_epi8(_mm512_xor_si512(bytes, bias_b), bias_b);
    } else {
      return byte_pair < Rdna3 && staged_clamp > (word, k, mask_b);
    }
  }

  V sum_b(V sum, V b) const { return _mm512_sub_epi16(sum, _mm512_maddubs_epi16(bias_a, b)); }
  static V bias_correction(V sum) { return sum; }

  static Accumulator accumulator(V c) {
    if constexpr (narrow_clamp) {
      __m512i base = _mm512_min_epi32(_mm512_max_epi32(c, _mm512_set1_epi32(INT32_MIN - INT16_MIN)),
                                      _mm512_set1_epi32(INT32_MAX - INT16_MAX));
      return {_mm512_sub_epi32(c, base), base};
    } else {
      return {c, V{}};
    }
  }
  static V result(Accumulator acc) {
    if constexpr (narrow_clamp) {
      __m512i offset = _mm512_srai_epi32(_mm512_slli_epi32(acc.value, 16), 16);
      return _mm512_add_epi32(acc.base, offset);
    } else {
      return acc.value;
    }
  }

  static V start_stage(Accumulator acc, V bias) {
    if constexpr (Bits == 4) {
      return bias;
    }
    return Clamp ? V{} : acc.value;
  }
  static V product(V sum, V a, V b) {
    if constexpr (Bits == 4) {
      return _mm512_add_epi16(sum, _mm512_maddubs_epi16(a, b));
    } else {
      return _mm512_add_epi32(sum, _mm512_madd_epi16(a, b));
    }
  }

  static void finish_stage(Accumulator &acc, V sum) {
    if constexpr (Bits == 4) {
      sum = _mm512_madd_epi16(sum, _mm512_set1_epi16(1));
    }
    if constexpr (Positive) {
      __m512i limit = _mm512_sub_epi32(_mm512_set1_epi32(INT32_MAX), sum);
      acc.value = _mm512_add_epi32(_mm512_min_epi32(acc.value, limit), sum);
    } else if constexpr (narrow_clamp) {
      acc.value = _mm512_adds_epi16(acc.value, sum);
    } else if constexpr (Clamp) {
      __m512i value = _mm512_add_epi32(acc.value, sum);
      __mmask16 ovf = _mm512_cmpgt_epi32_mask(acc.value, value) ^ _mm512_movepi32_mask(sum);
      __m512i limit =
          _mm512_xor_si512(_mm512_srai_epi32(acc.value, 31), _mm512_set1_epi32(INT32_MAX));
      acc.value = _mm512_mask_mov_epi32(value, ovf, limit);
    } else if constexpr (Bits == 4) {
      acc.value = _mm512_add_epi32(acc.value, sum);
    } else {
      acc.value = sum;
    }
  }

private:
  V bias_a, bias_b, mask_a, mask_b;
};

// Evaluate a validated dense integer instruction in its physical wave layout.
template <int Bits, int K, int Lanes = 32, bool Rdna3 = false>
void dense_integer(uint32_t flags, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *const *b, const uint32_t *const *c) {
  const auto run = [&](auto dot) {
    constexpr int rows = Bits == 8 && Rdna3 && decltype(dot)::staged_clamp ? 8 : 2;
    rocjitsu::arith::integer_wmma_tile<Bits, K, Lanes, Rdna3, rows>(dot, d, a, b, c);
  };
  if (flags == RJ_ARITH_WMMA_CLAMP) {
    run(Products<Bits, Rdna3, true, true>(flags));
  } else if (flags & RJ_ARITH_WMMA_CLAMP) {
    run(Products<Bits, Rdna3, true>(flags));
  } else {
    run(Products<Bits, Rdna3, false>(flags));
  }
}

} // namespace

namespace rocjitsu::arith {

// Compute a gfx12 integer tile in Wave32 layout; D may alias A/B/C.
// Requires validated modifiers and a supported compile-time shape.
template <int Bits, int K>
void integer_wmma_x86_64_v4(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *c) {
  static_assert((Bits == 8 && K == 16) || (Bits == 4 && (K == 16 || K == 32)));
  dense_integer<Bits, K>(modifiers, d, a, b, c);
}

template void integer_wmma_x86_64_v4<8, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_x86_64_v4<4, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_x86_64_v4<4, 32>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);

// Compute a gfx11 integer tile for the selected wave size and factor width.
// Requires validated modifiers; D may overlap source registers.
template <int Bits, int Lanes>
void integer_wmma_replicated_x86_64_v4(uint32_t modifiers, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  dense_integer<Bits, 16, Lanes, true>(modifiers, d, a, b, c);
}

template void integer_wmma_replicated_x86_64_v4<4, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v4<4, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v4<8, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v4<8, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
