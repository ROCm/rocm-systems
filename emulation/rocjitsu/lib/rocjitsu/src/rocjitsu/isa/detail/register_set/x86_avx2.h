// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_DETAIL_REGISTER_SET_X86_AVX2_H_
#define ROCJITSU_ISA_DETAIL_REGISTER_SET_X86_AVX2_H_

#include "rocjitsu/isa/detail/register_set/types.h"
#include <bit>
#include <cstring>
#include <immintrin.h>
#include <limits>

namespace rocjitsu::register_set_detail {

template <> struct WordOps<RegisterSetWordType::Avx2M256> {
  using Word = __m256i;
  static constexpr size_t kWordBits = 256;
  static Word word_all_bits() { return _mm256_set1_epi64x(-1); }
  static unsigned nonempty_lanes(Word word) {
    const Word empty = _mm256_cmpeq_epi64(word, _mm256_setzero_si256());
    return ~static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(empty))) & 15u;
  }
  static Word word_or(Word a, Word b) { return _mm256_or_si256(a, b); }
  static Word word_and(Word a, Word b) { return _mm256_and_si256(a, b); }
  static Word word_and_not(Word a, Word b) {
    // Computes a & ~b; the intrinsic complements its first operand.
    return _mm256_andnot_si256(b, a);
  }
  static bool word_none(Word a) { return _mm256_testz_si256(a, a) != 0; }
  static bool word_equal(Word a, Word b) {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) == -1;
  }
  static Word word_bit_mask(unsigned bitIndex) {
    // XOR selects one lane. All other shifts are >= 64 and produce zero.
    const Word shiftBits =
        _mm256_xor_si256(_mm256_set1_epi64x(bitIndex), _mm256_setr_epi64x(0, 64, 128, 192));
    return _mm256_sllv_epi64(_mm256_set1_epi64x(1), shiftBits);
  }
  static Word word_prefix(unsigned end) {
    const Word counts =
        _mm256_sub_epi64(_mm256_set1_epi64x(end), _mm256_set_epi64x(192, 128, 64, 0));
    const Word positive = _mm256_cmpgt_epi64(counts, _mm256_setzero_si256());
    const Word high = _mm256_sllv_epi64(word_all_bits(), counts);
    return _mm256_andnot_si256(high, positive);
  }
  static Word word_count_lanes(Word bits) {
    const Word table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2,
                                        1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const Word low_mask = _mm256_set1_epi8(15);
    const Word low = _mm256_shuffle_epi8(table, _mm256_and_si256(bits, low_mask));
    const Word high =
        _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits, 4), low_mask));
    return _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256());
  }
  static size_t sum_count_lanes(Word sums) {
    const __m128i halves =
        _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    const __m128i total = _mm_add_epi64(halves, _mm_unpackhi_epi64(halves, halves));
    return static_cast<size_t>(_mm_cvtsi128_si64(total));
  }
  template <size_t NumWords> static size_t count_bits(const Word (&words)[NumWords]) {
    // Keep counts in SIMD lanes across words, reducing to a scalar only once.
    Word numBitsPerLane = _mm256_setzero_si256();
    for (Word word : words)
      numBitsPerLane = _mm256_add_epi64(numBitsPerLane, word_count_lanes(word));
    return sum_count_lanes(numBitsPerLane);
  }
  // Reload membership after callbacks so edits to later indices are observed.
  template <size_t Words, typename Emit>
  static void for_each_index(const Word (&words)[Words], Emit emit) {
    for (size_t word = 0; word < Words; ++word) {
      unsigned lanes = nonempty_lanes(words[word]);
      while (lanes) {
        unsigned lane = std::countr_zero(lanes);
        // Scalar membership cursors avoid carrying a SIMD snapshot across
        // callbacks. memcpy reads the vector's representation without aliasing
        // violations and lowers to a scalar load for this fixed-size copy.
        const auto *address = reinterpret_cast<const unsigned char *>(&words[word]) + lane * 8;
        const auto load = [&] {
          uint64_t value;
          std::memcpy(&value, address, sizeof(value));
          return value;
        };
        uint64_t members = load(), remaining = members;
        // Full lanes emit consecutive indices. After an edit, resume set-bit
        // iteration strictly after the last emitted index, including at bit 63.
        unsigned cursor = 0;
        if (members == ~uint64_t{0}) {
#pragma GCC unroll 8
          for (; cursor < 64; ++cursor) {
            emit(static_cast<uint16_t>(word * 256 + lane * 64 + cursor));
            members = load();
            if (members != ~uint64_t{0}) {
              remaining = members & (~uint64_t{1} << cursor);
              break;
            }
          }
          if (cursor == 64)
            remaining = 0;
        }
        while (remaining) {
          unsigned bit = std::countr_zero(remaining);
          emit(static_cast<uint16_t>(word * 256 + lane * 64 + bit));
          // Keep edit recovery on a cold path. In particular, this prevents Clang
          // from putting its shift and mask on the cursor's dependency chain.
          const uint64_t updated = load();
          if (updated != members) [[unlikely]] {
            remaining = updated & (~uint64_t{1} << bit);
            members = updated;
          } else {
            remaining &= remaining - 1;
          }
        }
        // Observe additions to later, previously empty lanes as well as removals.
        lanes = nonempty_lanes(words[word]) & (~1u << lane);
      }
    }
  }
};

} // namespace rocjitsu::register_set_detail

#endif // ROCJITSU_ISA_DETAIL_REGISTER_SET_X86_AVX2_H_
