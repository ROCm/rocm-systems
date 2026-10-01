// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <bit>

namespace rocjitsu {

namespace {

using namespace register_set_detail;

// Visit a nonempty, in-bounds bit range. Stop when an operation returns a
// value other than kContinue; otherwise return kContinue after the last word.
// Full words need no range mask. Inline to retain the caller's capacity bounds.
template <typename WordType, size_t NumWords, typename Operation>
[[gnu::always_inline]] inline bool visit_range(WordType (&words)[NumWords], size_t startBit,
                                               size_t endBit, Operation operation) {
  // Prologue: initial partial word.
  size_t word = startBit / kWordBits;
  const size_t firstBitInWord = startBit % kWordBits;
  if (firstBitInWord != 0) {
    const size_t endBitInWord = std::min(kWordBits, endBit - word * kWordBits);
    const Word mask = word_range(firstBitInWord, endBitInWord);
    if (operation.partial_word(words[word], mask) != Operation::kContinue)
      return !Operation::kContinue;
    if (endBitInWord < kWordBits)
      return Operation::kContinue;
    ++word;
  }

  // Middle loop: full words.
  const size_t endWord = endBit / kWordBits;
  for (; word < endWord; ++word)
    if (operation.full_word(words[word]) != Operation::kContinue)
      return !Operation::kContinue;

  // Epilogue: final partial word.
  const size_t endBitInWord = endBit % kWordBits;
  if (endBitInWord != 0) {
    const Word mask = word_prefix(endBitInWord);
    return operation.partial_word(words[word], mask);
  }
  return Operation::kContinue;
}

struct ContainsWords {
  static constexpr bool kContinue = true;
  bool partial_word(const Word &word, Word mask) const {
    return word_none(word_and_not(mask, word));
  }
  bool full_word(const Word &word) const { return word_equal(word, word_all_bits()); }
};
struct IntersectsWords {
  static constexpr bool kContinue = false;
  bool partial_word(const Word &word, Word mask) const { return !word_none(word_and(word, mask)); }
  bool full_word(const Word &word) const { return !word_none(word); }
};

template <typename Update> struct UpdateWords {
  static constexpr bool kContinue = true;
  Update update;
  bool partial_word(Word &word, Word mask) const {
    word = update(word, mask);
    return true;
  }
  bool full_word(Word &word) const {
    word = update(word, word_all_bits());
    return true;
  }
};

template <size_t CapacityBits, size_t NumWords, typename Update>
void update_range(Word (&words)[NumWords], size_t startBit, size_t numBits, Update update) {
  if (numBits == 0)
    return;
  if (startBit >= CapacityBits)
    return;
  const size_t endBit = startBit + std::min(numBits, CapacityBits - startBit);
  visit_range(words, startBit, endBit, UpdateWords<Update>{update});
}
template <size_t CapacityBits, size_t NumWords>
void set_range(Word (&words)[NumWords], size_t startBit, size_t numBits) {
  if (numBits == 1) {
    if (startBit < CapacityBits)
      word_set_bit(words[startBit / kWordBits], startBit % kWordBits);
    return;
  }
  update_range<CapacityBits>(words, startBit, numBits, word_or);
}
template <size_t CapacityBits, size_t NumWords>
void reset_range(Word (&words)[NumWords], size_t startBit, size_t numBits) {
  if (numBits == 1) {
    if (startBit < CapacityBits)
      word_reset_bit(words[startBit / kWordBits], startBit % kWordBits);
    return;
  }
  update_range<CapacityBits>(words, startBit, numBits, word_and_not);
}
template <size_t CapacityBits, size_t NumWords>
bool contains_range(const Word (&words)[NumWords], size_t startBit, size_t numBits) {
  if (startBit >= CapacityBits || numBits > CapacityBits - startBit)
    return false;
  if (numBits == 1)
    return word_test_bit(words[startBit / kWordBits], startBit % kWordBits);
  if (numBits == 0)
    return true;
  const size_t endBit = startBit + numBits;

  return visit_range(words, startBit, endBit, ContainsWords{});
}
template <size_t CapacityBits, size_t NumWords>
bool intersects_range(const Word (&words)[NumWords], size_t startBit, size_t numBits) {
  if (numBits == 1)
    return startBit < CapacityBits &&
           word_test_bit(words[startBit / kWordBits], startBit % kWordBits);
  if (numBits == 0 || startBit >= CapacityBits)
    return false;
  const size_t endBit = startBit + std::min(numBits, CapacityBits - startBit);

  return visit_range(words, startBit, endBit, IntersectsWords{});
}
template <size_t NumWords, typename Operation>
void combine(Word (&lhs)[NumWords], const Word (&rhs)[NumWords], Operation operation) {
  for (size_t word = 0; word < NumWords; ++word)
    lhs[word] = operation(lhs[word], rhs[word]);
}
template <size_t NumWords> bool bits_none(const Word (&words)[NumWords]) {
  for (const Word word : words)
    if (!word_none(word))
      return false;
  return true;
}
template <size_t NumWords> size_t count_bits(const Word (&words)[NumWords]) {
#if defined(__AVX2__)
  // Keep counts in SIMD lanes across words, reducing to a scalar only once.
  Word numBitsPerLane = _mm256_setzero_si256();
  for (const Word word : words)
    numBitsPerLane = _mm256_add_epi64(numBitsPerLane, word_count_lanes(word));
  return sum_count_lanes(numBitsPerLane);
#else
  size_t numBits = 0;
  for (const Word word : words)
    numBits += word_count(word);
  return numBits;
#endif
}
template <size_t NumWords>
bool intersects_bits(const Word (&lhs)[NumWords], const Word (&rhs)[NumWords]) {
  for (size_t word = 0; word < NumWords; ++word)
    if (!word_none(word_and(lhs[word], rhs[word])))
      return true;
  return false;
}

} // namespace

void RegisterSet::expand(RegisterRef ref) {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    set_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
    break;
  case RegClass::VGPR:
    set_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
    break;
  case RegClass::ACC_VGPR:
    set_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
    break;
  case RegClass::TTMP:
    // Trap temporaries are known to the ISA but not tracked by this set: no
    // bitset and no mask bit, so index/width are dropped (as they are on the
    // general def/use path). Explicit arm keeps the switch exhaustive.
    break;
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    // Special singleton: index/width are meaningless, so just set its bit.
    special_regs_ |= special_bit(ref.cls);
    break;
  }
}

void RegisterSet::erase(RegisterRef ref) {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    reset_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
    break;
  case RegClass::VGPR:
    reset_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
    break;
  case RegClass::ACC_VGPR:
    reset_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
    break;
  case RegClass::TTMP: // untracked: nothing to clear
    break;
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    special_regs_ &= static_cast<uint16_t>(~special_bit(ref.cls));
    break;
  }
}

void RegisterSet::clear_class(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    std::fill(std::begin(sgprs_), std::end(sgprs_), Word{});
    break;
  case RegClass::VGPR:
    std::fill(std::begin(vgprs_), std::end(vgprs_), Word{});
    break;
  case RegClass::ACC_VGPR:
    std::fill(std::begin(acc_vgprs_), std::end(acc_vgprs_), Word{});
    break;
  case RegClass::TTMP: // untracked: nothing to clear
    break;
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    special_regs_ &= static_cast<uint16_t>(~special_bit(cls));
    break;
  }
}

bool RegisterSet::contains(RegisterRef ref) const {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return contains_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
  case RegClass::VGPR:
    return contains_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
  case RegClass::ACC_VGPR:
    return contains_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
  case RegClass::TTMP: // untracked: never present
    return false;
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    return (special_regs_ & special_bit(ref.cls)) != 0;
  }
  return false; // unreachable for a valid RegClass; a new class trips -Wswitch first
}

bool RegisterSet::none() const {
  return bits_none(sgprs_) && bits_none(vgprs_) && bits_none(acc_vgprs_) && special_regs_ == 0;
}

size_t RegisterSet::size() const {
  return ordinary_size() + static_cast<size_t>(std::popcount(special_regs_));
}

size_t RegisterSet::ordinary_size() const {
  return count_bits(sgprs_) + count_bits(vgprs_) + count_bits(acc_vgprs_);
}

bool RegisterSet::intersects(RegisterRef ref) const {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return intersects_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
  case RegClass::VGPR:
    return intersects_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
  case RegClass::ACC_VGPR:
    return intersects_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
  default:
    return false;
  }
}

bool RegisterSet::intersects(const RegisterSet &rhs) const {
  return intersects_bits(sgprs_, rhs.sgprs_) || intersects_bits(vgprs_, rhs.vgprs_) ||
         intersects_bits(acc_vgprs_, rhs.acc_vgprs_) || (special_regs_ & rhs.special_regs_) != 0;
}

RegisterSet &RegisterSet::operator|=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, word_or);
  combine(vgprs_, rhs.vgprs_, word_or);
  combine(acc_vgprs_, rhs.acc_vgprs_, word_or);
  special_regs_ |= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator&=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, word_and);
  combine(vgprs_, rhs.vgprs_, word_and);
  combine(acc_vgprs_, rhs.acc_vgprs_, word_and);
  special_regs_ &= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator-=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, word_and_not);
  combine(vgprs_, rhs.vgprs_, word_and_not);
  combine(acc_vgprs_, rhs.acc_vgprs_, word_and_not);
  special_regs_ &= static_cast<uint16_t>(~rhs.special_regs_);
  return *this;
}

} // namespace rocjitsu
