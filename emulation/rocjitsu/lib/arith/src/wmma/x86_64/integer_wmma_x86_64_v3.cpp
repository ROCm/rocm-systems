// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 operations and dispatch for the shared dense INT8/INT4 WMMA algorithm.
// Uses word pairs for bytes and packed byte quartets for nibbles.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/integer_wmma.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Select two bytes within each packed word and duplicate each to a halfword.
__m256i duplicate_bytes(__m256i words, int indices) {
  auto offsets = _mm256_setr_epi32(0, 0x04040404, 0x08080808, 0x0c0c0c0c, 0, 0x04040404, 0x08080808,
                                   0x0c0c0c0c);
  return _mm256_shuffle_epi8(words, _mm256_add_epi8(_mm256_set1_epi32(indices), offsets));
}

// Pack four nibbles from one hardware stage into bytes for VPMADDUBSW.
template <bool Adjacent> __m256i packed_nibbles(__m256i words, int k) {
  // If the whole packed word belongs to one stage, multiply all even nibbles
  // and then all odd nibbles. Each is already in its own byte.
  if constexpr (!Adjacent) {
    return _mm256_and_si256(k & 4 ? _mm256_srli_epi16(words, 4) : words, _mm256_set1_epi8(15));
  }
  auto bytes = duplicate_bytes(words, k & 4 ? 0x03030202 : 0x01010000);
  auto nibbles = _mm256_blendv_epi8(bytes, _mm256_srli_epi16(bytes, 4), _mm256_set1_epi16(-256));
  return _mm256_and_si256(nibbles, _mm256_set1_epi8(15));
}

// Gfx11 clamped IU8 consumes adjacent byte pairs. Otherwise interleave the
// two pairs of each word as (0,2) and (1,3) to widen in place.
template <bool Adjacent> __m256i byte_pair(__m256i words, int k, __m256i mask) {
  if constexpr (Adjacent) {
    words = duplicate_bytes(words, k & 2 ? 0x03030202 : 0x01010000);
  } else if (!(k & 2)) {
    words = _mm256_slli_epi16(words, 8);
  }
  return _mm256_and_si256(_mm256_srai_epi16(words, 8), mask);
}

// Word-pair dots cover the full IU8 range. IU4 quartet dots accumulate exact
// halfword partials, including the signed-A column correction, until stage end.
template <int Bits, bool Rdna3, bool Clamp, bool Positive = false> class Products {
public:
  using V = __m256i;
  static constexpr int width = 8;
  static constexpr int products = Bits == 4 ? 4 : 2;
  static constexpr bool biased = Bits == 4;
  static constexpr bool staged_clamp = Clamp && !Positive;
  static constexpr bool narrow_clamp = Bits == 4 && staged_clamp;
  struct Accumulator {
    V value, base;
  };

  explicit Products(uint32_t flags) {
    bool signed_a = flags & RJ_ARITH_WMMA_SIGNED_A;
    bool signed_b = flags & RJ_ARITH_WMMA_SIGNED_B;
    bias_a = _mm256_set1_epi8(signed_a ? 8 : 0);
    bias_b = _mm256_set1_epi8(signed_b ? 8 : 0);
    mask_a = _mm256_set1_epi16(signed_a ? -1 : 255);
    mask_b = _mm256_set1_epi16(signed_b ? -1 : 255);
  }

  // Unaligned register transfers and single packed-factor broadcasts.
  static V load(const uint32_t *p) { return _mm256_loadu_si256(reinterpret_cast<const V *>(p)); }
  static void store(uint32_t *p, V v) { _mm256_storeu_si256(reinterpret_cast<V *>(p), v); }
  static V broadcast(uint32_t word) { return _mm256_set1_epi32(word); }

  // Byte products take unsigned A. Signed nibbles map to [0,15] by adding eight.
  V unpack_a(V word, int k) const {
    if constexpr (Bits == 4) {
      return _mm256_xor_si256(packed_nibbles < Rdna3 && staged_clamp > (word, k), bias_a);
    } else {
      return byte_pair < Rdna3 && staged_clamp > (word, k, mask_a);
    }
  }

  // Sign- or zero-extend B for the native word or byte dot.
  V unpack_b(V word, int k) const {
    if constexpr (Bits == 4) {
      auto bytes = packed_nibbles < Rdna3 && staged_clamp > (word, k);
      return _mm256_sub_epi8(_mm256_xor_si256(bytes, bias_b), bias_b);
    } else {
      return byte_pair < Rdna3 && staged_clamp > (word, k, mask_b);
    }
  }

  // Accumulate the negative IU4 bias directly in each pair of halfword partials.
  V sum_b(V sum, V b) const { return _mm256_sub_epi16(sum, _mm256_maddubs_epi16(bias_a, b)); }
  static V bias_correction(V sum) { return sum; }

  // The total absolute IU4 movement is at most 32 * 15^2. Choose a base
  // near either signed limit so saturating the low halfword clips at exactly
  // that limit. Away from the limits the initial offset is zero.
  static Accumulator accumulator(V c) {
    if constexpr (narrow_clamp) {
      auto base = _mm256_min_epi32(_mm256_max_epi32(c, _mm256_set1_epi32(INT32_MIN - INT16_MIN)),
                                   _mm256_set1_epi32(INT32_MAX - INT16_MAX));
      return {_mm256_sub_epi32(c, base), base};
    } else {
      return {c, V{}};
    }
  }
  static V result(Accumulator acc) {
    if constexpr (narrow_clamp) {
      auto offset = _mm256_srai_epi32(_mm256_slli_epi32(acc.value, 16), 16);
      return _mm256_add_epi32(acc.base, offset);
    } else {
      return acc.value;
    }
  }

  // IU4 delays the horizontal halfword addition until the end of the stage.
  static V start_stage(Accumulator acc, V bias) {
    if constexpr (Bits == 4) {
      return bias;
    }
    return Clamp ? V{} : acc.value;
  }
  static V product(V sum, V a, V b) {
    if constexpr (Bits == 4) {
      return _mm256_add_epi16(sum, _mm256_maddubs_epi16(a, b));
    } else {
      return _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
    }
  }

  // All-unsigned factors can only move C upward. Their final positive
  // saturation equals clamping after every hardware stage.
  static void finish_stage(Accumulator &acc, V sum) {
    if constexpr (Bits == 4) {
      sum = _mm256_madd_epi16(sum, _mm256_set1_epi16(1));
    }
    if constexpr (Positive) {
      auto limit = _mm256_sub_epi32(_mm256_set1_epi32(INT32_MAX), sum);
      acc.value = _mm256_add_epi32(_mm256_min_epi32(acc.value, limit), sum);
    } else if constexpr (narrow_clamp) {
      acc.value = _mm256_adds_epi16(acc.value, sum);
    } else if constexpr (Clamp) {
      auto value = _mm256_add_epi32(acc.value, sum);
      // A signed add overflowed when its ordering disagrees with the addend's sign.
      auto overflow = _mm256_xor_si256(_mm256_cmpgt_epi32(acc.value, value), sum);
      auto limit = _mm256_xor_si256(_mm256_srai_epi32(acc.value, 31), _mm256_set1_epi32(INT32_MAX));
      acc.value = _mm256_castps_si256(_mm256_blendv_ps(
          _mm256_castsi256_ps(value), _mm256_castsi256_ps(limit), _mm256_castsi256_ps(overflow)));
    } else if constexpr (Bits == 4) {
      acc.value = _mm256_add_epi32(acc.value, sum);
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
    constexpr int rows = Rdna3 && decltype(dot)::staged_clamp ? 4 : 1;
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
void integer_wmma_x86_64_v3(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *c) {
  static_assert((Bits == 8 && K == 16) || (Bits == 4 && (K == 16 || K == 32)));
  dense_integer<Bits, K>(modifiers, d, a, b, c);
}

template void integer_wmma_x86_64_v3<8, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_x86_64_v3<4, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_x86_64_v3<4, 32>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                            const uint32_t *const *, const uint32_t *const *);

// Compute a gfx11 integer tile for the selected wave size and factor width.
// Requires validated modifiers; D may overlap source registers.
template <int Bits, int Lanes>
void integer_wmma_replicated_x86_64_v3(uint32_t modifiers, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  dense_integer<Bits, 16, Lanes, true>(modifiers, d, a, b, c);
}

template void integer_wmma_replicated_x86_64_v3<4, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<4, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<8, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<8, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
