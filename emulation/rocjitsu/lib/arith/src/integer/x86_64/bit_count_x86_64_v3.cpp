// SPDX-License-Identifier: MIT

// AVX2 population count, bit reversal, and first/last significant-bit searches.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/bit_count.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Return a population count for each integer lane.
__m256i population(__m256i x) {
  auto table =
      _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4));
  auto nibble = _mm256_set1_epi8(15);
  auto low = _mm256_shuffle_epi8(table, _mm256_and_si256(x, nibble));
  auto high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(x, 4), nibble));
  auto pairs = _mm256_maddubs_epi16(_mm256_add_epi8(low, high), _mm256_set1_epi8(1));
  return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

} // namespace

namespace rocjitsu::arith {

// Count or reorder bits. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <BitCount Op, int Lanes>
void bit_count_x86_64_v3(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b) {
  static_assert(Op != BitCount::MaskedHigh || Lanes != 32);
  for (int lane = 0; lane < Lanes; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    if constexpr (Op == BitCount::Sign) {
      x = _mm256_xor_si256(x, _mm256_srai_epi32(x, 31));
    }
    __m256i result;
    if constexpr (Op == BitCount::Leading || Op == BitCount::Sign || Op == BitCount::Trailing) {
      auto value = x;
      if constexpr (Op == BitCount::Trailing) {
        value = _mm256_sub_epi32(_mm256_and_si256(x, _mm256_sub_epi32(_mm256_setzero_si256(), x)),
                                 _mm256_set1_epi32(1));
      } else {
        // rocjitsu clz_u32_simd: propagate the highest bit downward.
        value = _mm256_or_si256(value, _mm256_srli_epi32(value, 1));
        value = _mm256_or_si256(value, _mm256_srli_epi32(value, 2));
        value = _mm256_or_si256(value, _mm256_srli_epi32(value, 4));
        value = _mm256_or_si256(value, _mm256_srli_epi32(value, 8));
        value = _mm256_or_si256(value, _mm256_srli_epi32(value, 16));
        value = _mm256_xor_si256(value, _mm256_set1_epi32(-1));
      }
      result = _mm256_or_si256(population(value), _mm256_cmpeq_epi32(x, _mm256_setzero_si256()));
    } else {
      if constexpr (Op == BitCount::MaskedLow || Op == BitCount::MaskedHigh) {
        auto index =
            _mm256_add_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(lane));
        if constexpr (Op == BitCount::MaskedHigh) {
          index = _mm256_max_epi32(_mm256_sub_epi32(index, _mm256_set1_epi32(32)),
                                   _mm256_setzero_si256());
        }
        auto below =
            _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), index), _mm256_set1_epi32(1));
        x = _mm256_and_si256(x, below);
      }
      auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      result = _mm256_add_epi32(population(x), y);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void bit_count_x86_64_v3<BitCount::Leading, 32>(uint32_t exec_mask, uint32_t *d,
                                                         const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Trailing, 32>(uint32_t exec_mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Sign, 32>(uint32_t exec_mask, uint32_t *d,
                                                      const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Population, 32>(uint32_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::MaskedLow, 32>(uint32_t exec_mask, uint32_t *d,
                                                           const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::MaskedLow, 64>(uint64_t exec_mask, uint32_t *d,
                                                           const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::MaskedHigh, 64>(uint64_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);

template void bit_count_x86_64_v3<BitCount::Leading, 64>(uint64_t exec_mask, uint32_t *d,
                                                         const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Trailing, 64>(uint64_t exec_mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Sign, 64>(uint64_t exec_mask, uint32_t *d,
                                                      const uint32_t *a, const uint32_t *b);
template void bit_count_x86_64_v3<BitCount::Population, 64>(uint64_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
