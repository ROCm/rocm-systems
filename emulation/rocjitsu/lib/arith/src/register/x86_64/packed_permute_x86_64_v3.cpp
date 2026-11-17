// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 packed element permutations with masked output and VGPR alias support.
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
  for (int lane = 0; lane < lanes; lane += 8) {
    __m256i table[5], result[4], selectors[2];
    for (int word = 0; word < 5; ++word) {
      table[word] = _mm256_setzero_si256();
    }
    for (int word = 0; word < words; ++word) {
      auto p = word < low_words ? b[word] : a[word - low_words];
      table[word] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + lane));
      result[word] = _mm256_setzero_si256();
    }
    for (int word = 0; word < 2; ++word) {
      selectors[word] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[word] + lane));
    }
    for (int element = 0; element < 16; ++element) {
      auto index = _mm256_and_si256(
          _mm256_srl_epi32(selectors[element / 8], _mm_cvtsi32_si128((element % 8) * 4)),
          _mm256_set1_epi32(15));
      auto bit = _mm256_mullo_epi32(index, _mm256_set1_epi32(bits));
      auto word = _mm256_srli_epi32(bit, 5);
      auto shift = _mm256_and_si256(bit, _mm256_set1_epi32(31));
      auto low = _mm256_setzero_si256(), high = _mm256_setzero_si256();
      for (int i = 0; i < words; ++i) {
        auto selected = _mm256_cmpeq_epi32(word, _mm256_set1_epi32(i));
        low = _mm256_or_si256(low, _mm256_and_si256(selected, table[i]));
        high = _mm256_or_si256(high, _mm256_and_si256(selected, table[i + 1]));
      }
      auto value =
          _mm256_or_si256(_mm256_srlv_epi32(low, shift),
                          _mm256_sllv_epi32(high, _mm256_sub_epi32(_mm256_set1_epi32(32), shift)));
      value = _mm256_and_si256(value, _mm256_set1_epi32((1 << bits) - 1));
      int output = element * bits, target = output / 32, offset = output % 32;
      result[target] =
          _mm256_or_si256(result[target], _mm256_sll_epi32(value, _mm_cvtsi32_si128(offset)));
      if (offset + bits > 32) {
        result[target + 1] = _mm256_or_si256(
            result[target + 1], _mm256_srl_epi32(value, _mm_cvtsi32_si128(32 - offset)));
      }
    }
    for (int word = 0; word < words; ++word) {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[word] + lane), active, result[word]);
    }
  }
}
} // namespace

// Select the format specialization after baseline validation.
void packed_permute_x86_64_v3(int bits, int lanes, uint64_t exec_mask, uint32_t *const *d,
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
