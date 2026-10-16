// SPDX-License-Identifier: MIT

// AVX2 bitfield extraction/insertion and alignment of adjacent source words.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/bitfield.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Extract, insert, or align bitfields. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Bitfield Op>
void bitfield_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    __m256i x;
    x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i y;
    if constexpr (Op != Bitfield::Reverse) {
      y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    }
    __m256i z;
    if constexpr (Op != Bitfield::Reverse && Op != Bitfield::Mask) {
      z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    }
    __m256i result;
    if constexpr (Op == Bitfield::ExtractUnsigned || Op == Bitfield::ExtractSigned) {
      auto offset = _mm256_and_si256(y, _mm256_set1_epi32(31)),
           width = _mm256_and_si256(z, _mm256_set1_epi32(31));
      auto power = _mm256_sllv_epi32(_mm256_set1_epi32(1), width);
      auto field_mask = _mm256_sub_epi32(power, _mm256_set1_epi32(1));
      if constexpr (Op == Bitfield::ExtractUnsigned) {
        result = _mm256_and_si256(_mm256_srlv_epi32(x, offset), field_mask);
      } else {
        auto field = _mm256_and_si256(_mm256_srav_epi32(x, offset), field_mask);
        auto sign = _mm256_srli_epi32(power, 1);
        result = _mm256_sub_epi32(_mm256_xor_si256(field, sign), sign);
      }
    }
    if constexpr (Op == Bitfield::AlignBit || Op == Bitfield::AlignByte) {
      auto shift = _mm256_and_si256(z, _mm256_set1_epi32(Op == Bitfield::AlignBit ? 31 : 3));
      if constexpr (Op == Bitfield::AlignByte) {
        shift = _mm256_slli_epi32(shift, 3);
      }
      auto inverse = _mm256_sub_epi32(_mm256_set1_epi32(32), shift);
      result = _mm256_or_si256(_mm256_srlv_epi32(y, shift), _mm256_sllv_epi32(x, inverse));
    }
    if constexpr (Op == Bitfield::Permute) {
      // PSHUFB indices stay within each lane's four bytes. A small lookup
      // redirects selectors 8..11 to the bytes providing sign-fill bits.
      auto selector = _mm256_min_epu8(z, _mm256_set1_epi8(13));
      auto mapping = _mm256_broadcastsi128_si256(
          _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 7, 0, 0, 0, 0));
      auto source = _mm256_shuffle_epi8(mapping, selector);
      auto base = _mm256_setr_epi32(0, 0x04040404, 0x08080808, 0x0c0c0c0c, 0, 0x04040404,
                                    0x08080808, 0x0c0c0c0c);
      auto index = _mm256_or_si256(base, _mm256_and_si256(source, _mm256_set1_epi8(3)));
      auto from_a =
          _mm256_cmpeq_epi8(_mm256_and_si256(source, _mm256_set1_epi8(4)), _mm256_set1_epi8(4));
      auto value =
          _mm256_blendv_epi8(_mm256_shuffle_epi8(y, index), _mm256_shuffle_epi8(x, index), from_a);
      auto sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), value);
      value = _mm256_blendv_epi8(value, sign, _mm256_cmpgt_epi8(selector, _mm256_set1_epi8(7)));
      value = _mm256_andnot_si256(_mm256_cmpeq_epi8(selector, _mm256_set1_epi8(12)), value);
      result = _mm256_or_si256(value, _mm256_cmpeq_epi8(selector, _mm256_set1_epi8(13)));
    }
    if constexpr (Op == Bitfield::Insert) {
      result = _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z));
    }
    if constexpr (Op == Bitfield::Mask) {
      result = _mm256_sllv_epi32(
          _mm256_sub_epi32(
              _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_and_si256(x, _mm256_set1_epi32(31))),
              _mm256_set1_epi32(1)),
          _mm256_and_si256(y, _mm256_set1_epi32(31)));
    }
    if constexpr (Op == Bitfield::Reverse) {
      // Reverse each nibble with a byte lookup, then reverse the four bytes.
      auto table = _mm256_broadcastsi128_si256(
          _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15));
      auto nibble = _mm256_set1_epi8(15);
      auto low = _mm256_shuffle_epi8(table, _mm256_and_si256(x, nibble));
      auto high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(x, 4), nibble));
      result = _mm256_shuffle_epi8(_mm256_or_si256(_mm256_slli_epi16(low, 4), high),
                                   _mm256_broadcastsi128_si256(_mm_setr_epi8(
                                       3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12)));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void bitfield_x86_64_v3<Bitfield::ExtractUnsigned>(uint32_t exec_mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b,
                                                            const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::ExtractSigned>(uint32_t exec_mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b,
                                                          const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::Mask>(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                                                 const uint32_t *b, const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::Reverse>(uint32_t exec_mask, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b,
                                                    const uint32_t *c);

template void bitfield_x86_64_v3<Bitfield::AlignBit>(uint32_t, uint32_t *, const uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void bitfield_x86_64_v3<Bitfield::AlignByte>(uint32_t, uint32_t *, const uint32_t *,
                                                      const uint32_t *, const uint32_t *);

template void bitfield_x86_64_v3<Bitfield::Permute>(uint32_t, uint32_t *, const uint32_t *,
                                                    const uint32_t *, const uint32_t *);

template void bitfield_x86_64_v3<Bitfield::Insert>(uint32_t, uint32_t *, const uint32_t *,
                                                   const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
