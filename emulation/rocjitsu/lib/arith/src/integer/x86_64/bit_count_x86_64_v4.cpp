// SPDX-License-Identifier: MIT

// AVX-512 population count, bit reversal, and first/last significant-bit searches.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/bit_count.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Return a population count for each integer lane.
__m512i population(__m512i x) {
  auto table =
      _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4));
  auto nibble = _mm512_set1_epi8(15);
  auto low = _mm512_shuffle_epi8(table, _mm512_and_si512(x, nibble));
  auto high = _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16(x, 4), nibble));
  auto pairs = _mm512_maddubs_epi16(_mm512_add_epi8(low, high), _mm512_set1_epi8(1));
  return _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
}

} // namespace

namespace rocjitsu::arith {

// Count or reorder bits. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <BitCount Op, int Lanes>
void bit_count_x86_64_v4(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b) {
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    if constexpr (Op == BitCount::Sign) {
      x = _mm512_xor_si512(x, _mm512_srai_epi32(x, 31));
    }
    __m512i result;
    if constexpr (Op == BitCount::Leading || Op == BitCount::Sign || Op == BitCount::Trailing) {
      if constexpr (Op == BitCount::Trailing) {
        result = _mm512_sub_epi32(
            _mm512_set1_epi32(31),
            _mm512_lzcnt_epi32(_mm512_and_si512(x, _mm512_sub_epi32(_mm512_setzero_si512(), x))));
      } else {
        result = _mm512_mask_mov_epi32(_mm512_lzcnt_epi32(x),
                                       _mm512_cmpeq_epi32_mask(x, _mm512_setzero_si512()),
                                       _mm512_set1_epi32(-1));
      }
    } else {
      if constexpr (Op == BitCount::MaskedLow || Op == BitCount::MaskedHigh) {
        auto index = _mm512_add_epi32(
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm512_set1_epi32(lane));
        if constexpr (Op == BitCount::MaskedHigh) {
          index = _mm512_max_epi32(_mm512_sub_epi32(index, _mm512_set1_epi32(32)),
                                   _mm512_setzero_si512());
        }
        auto below =
            _mm512_sub_epi32(_mm512_sllv_epi32(_mm512_set1_epi32(1), index), _mm512_set1_epi32(1));
        x = _mm512_and_si512(x, below);
      }
      auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      result = _mm512_add_epi32(population(x), y);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void bit_count_x86_64_v4<BitCount::Leading, 32>(uint32_t exec_mask, uint32_t *d,
                                                         const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Trailing, 32>(uint32_t exec_mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Sign, 32>(uint32_t exec_mask, uint32_t *d,
                                                      const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Population, 32>(uint32_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::MaskedLow, 32>(uint32_t exec_mask, uint32_t *d,
                                                           const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::MaskedHigh, 32>(uint32_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::MaskedLow, 64>(uint64_t exec_mask, uint32_t *d,
                                                           const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::MaskedHigh, 64>(uint64_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);

template void bit_count_x86_64_v4<BitCount::Leading, 64>(uint64_t exec_mask, uint32_t *d,
                                                         const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Trailing, 64>(uint64_t exec_mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Sign, 64>(uint64_t exec_mask, uint32_t *d,
                                                      const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v4<BitCount::Population, 64>(uint64_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
