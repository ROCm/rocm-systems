// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_DETAIL_REGISTER_SET_PORTABLE_H_
#define ROCJITSU_ISA_DETAIL_REGISTER_SET_PORTABLE_H_

#include "rocjitsu/isa/detail/register_set/types.h"
#include <bit>
#include <limits>

namespace rocjitsu::register_set_detail {

template <> struct WordOps<RegisterSetWordType::StandardUint64> {
  using Word = uint64_t;
  static constexpr size_t kWordBits = std::numeric_limits<Word>::digits;
  static Word word_all_bits() { return std::numeric_limits<Word>::max(); }
  static Word word_or(Word a, Word b) { return a | b; }
  static Word word_and(Word a, Word b) { return a & b; }
  static Word word_and_not(Word a, Word b) { return a & ~b; }
  static bool word_none(Word a) { return a == 0; }
  static bool word_equal(Word a, Word b) { return a == b; }
  static Word word_bit_mask(unsigned bitIndex) { return Word{1} << bitIndex; }
  // A prefix of [0, end), including the empty and full-word cases.
  static Word word_prefix(unsigned end) {
    return end == kWordBits ? word_all_bits() : (Word{1} << end) - 1;
  }
  template <size_t NumWords> static size_t count_bits(const Word (&words)[NumWords]) {
    size_t count = 0;
    for (Word word : words)
      count += std::popcount(word);
    return count;
  }
  // Reload membership after callbacks so edits to later indices are observed.
  template <size_t Words, typename Emit>
  static void for_each_index(const Word (&words)[Words], Emit emit) {
    for (size_t word = 0; word < Words; ++word) {
      Word members = words[word];
      Word remaining = members;
      while (remaining) {
        unsigned bit = std::countr_zero(remaining);
        emit(static_cast<uint16_t>(word * kWordBits + bit));
        Word updated = words[word];
        if (updated == members)
          remaining &= remaining - 1;
        else
          remaining = bit == kWordBits - 1 ? 0 : updated & (word_all_bits() << (bit + 1));
        members = updated;
      }
    }
  }
};

} // namespace rocjitsu::register_set_detail

#endif // ROCJITSU_ISA_DETAIL_REGISTER_SET_PORTABLE_H_
