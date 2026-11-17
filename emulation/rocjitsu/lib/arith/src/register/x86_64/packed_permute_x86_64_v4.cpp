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
  for (int lane = 0; lane < lanes; lane += 16) {
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
