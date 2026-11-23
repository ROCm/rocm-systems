// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 packed element permutations with masked output and VGPR alias support.
#include "register/packed_permute.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
// Keep variable source shifts within words; the following word supplies any
// crossing bits. A zero sentinel supplies the end of the packed table.
template <int Bits>
void permute(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int bits = Bits, words = Bits / 2, low_words = Bits == 4 ? 1 : 2;
  if constexpr (Bits == 8) {
    const auto transpose = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (int lane = 0; lane < lanes; lane += 4) {
      auto mask = __mmask8((exec_mask >> lane) & 15);
      if (!mask) {
        continue;
      }
      auto load = [&](const uint32_t *p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + lane));
      };
      auto words = _mm512_castsi128_si512(load(b[0]));
      words = _mm512_inserti32x4(words, load(b[1]), 1);
      words = _mm512_inserti32x4(words, load(a[0]), 2);
      words = _mm512_inserti32x4(words, load(a[1]), 3);
      auto table = _mm512_permutexvar_epi32(transpose, words);
      auto low = load(c[0]), high = load(c[1]);
      auto packed = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi32(low, high)),
                                            _mm_unpackhi_epi32(low, high), 1);
      auto pairs = _mm512_cvtepu8_epi16(packed);
      auto index =
          _mm512_or_si512(_mm512_and_si512(pairs, _mm512_set1_epi16(15)),
                          _mm512_slli_epi16(_mm512_and_si512(pairs, _mm512_set1_epi16(240)), 4));
      auto out = _mm512_permutexvar_epi32(transpose, _mm512_shuffle_epi8(table, index));
      _mm_mask_storeu_epi32(d[0] + lane, mask, _mm512_castsi512_si128(out));
      _mm_mask_storeu_epi32(d[1] + lane, mask, _mm512_extracti32x4_epi32(out, 1));
      _mm_mask_storeu_epi32(d[2] + lane, mask, _mm512_extracti32x4_epi32(out, 2));
      _mm_mask_storeu_epi32(d[3] + lane, mask, _mm512_extracti32x4_epi32(out, 3));
    }
    return;
  }
  for (int lane = 0; lane < lanes; lane += 16) {
    if (uint16_t(exec_mask >> lane) == 0) {
      continue;
    }
    __m512i table[5], result[4], selectors[2];
    for (int word = 0; word < 5; ++word) {
      table[word] = _mm512_setzero_si512();
    }
    for (int word = 0; word < words; ++word) {
      auto p = word < low_words ? b[word] : a[word - low_words];
      table[word] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(p + lane));
      result[word] = _mm512_setzero_si512();
    }
    for (int word = 0; word < 2; ++word) {
      selectors[word] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c[word] + lane));
    }
    for (int element = 0; element < 16; ++element) {
      auto index = _mm512_and_si512(
          _mm512_srl_epi32(selectors[element / 8], _mm_cvtsi32_si128((element % 8) * 4)),
          _mm512_set1_epi32(15));
      auto bit = _mm512_mullo_epi32(index, _mm512_set1_epi32(bits));
      auto word = _mm512_srli_epi32(bit, 5);
      auto shift = _mm512_and_si512(bit, _mm512_set1_epi32(31));
      auto low = _mm512_setzero_si512(), high = _mm512_setzero_si512();
      for (int i = 0; i < words; ++i) {
        auto selected = _mm512_cmpeq_epi32_mask(word, _mm512_set1_epi32(i));
        low = _mm512_mask_mov_epi32(low, selected, table[i]);
        high = _mm512_mask_mov_epi32(high, selected, table[i + 1]);
      }
      auto value =
          _mm512_or_si512(_mm512_srlv_epi32(low, shift),
                          _mm512_sllv_epi32(high, _mm512_sub_epi32(_mm512_set1_epi32(32), shift)));
      value = _mm512_and_si512(value, _mm512_set1_epi32((1 << bits) - 1));
      int output = element * bits, target = output / 32, offset = output % 32;
      result[target] =
          _mm512_or_si512(result[target], _mm512_sll_epi32(value, _mm_cvtsi32_si128(offset)));
      if (offset + bits > 32) {
        result[target + 1] = _mm512_or_si512(
            result[target + 1], _mm512_srl_epi32(value, _mm_cvtsi32_si128(32 - offset)));
      }
    }
    for (int word = 0; word < words; ++word) {
      _mm512_mask_storeu_epi32(d[word] + lane, __mmask16(exec_mask >> lane), result[word]);
    }
  }
}
} // namespace

// Select the format specialization after baseline validation.
void packed_permute_x86_64_v4(int bits, int lanes, uint64_t exec_mask, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  switch (bits) {
  case 4:
    permute<4>(lanes, exec_mask, d, a, b, c);
    break;
  case 6:
    permute<6>(lanes, exec_mask, d, a, b, c);
    break;
  case 8:
    permute<8>(lanes, exec_mask, d, a, b, c);
    break;
  }
}
} // namespace rocjitsu::arith
