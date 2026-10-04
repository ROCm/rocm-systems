// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_set.h
/// @brief Register references and register sets for ISA-level register files.
///
/// @details Tracks the ordinary indexed register files (SGPR, VGPR, AccVGPR) as
/// per-index bitsets and the architectural special registers (EXEC, VCC, SCC,
/// M0, FLAT_SCRATCH, PC) as a compact singleton mask in the same set.
/// Consumers that only reason about allocatable/indexed registers (scratch
/// liveness, spilling) project the special members out with `ordinary_only()`.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace rocjitsu {

/// @brief RegisterSet storage capacities, derived from AMDGPU family traits.
///
/// @details These are storage bounds for an ISA-independent analysis set, not
/// per-kernel allocation limits. Wave32 vs. Wave64 changes lane count, not the
/// number of SGPR/VGPR indices addressable within a wavefront register file.
inline constexpr size_t REGISTER_SET_MAX_SGPRS =
    std::max<size_t>(amdgpu::CdnaIsaBase::MAX_SGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);
inline constexpr size_t REGISTER_SET_MAX_VGPRS = MAX_SUPPORTED_ADDRESSABLE_VGPRS_PER_WF;
inline constexpr size_t REGISTER_SET_MAX_ACC_VGPRS = REGISTER_SET_MAX_VGPRS;
inline constexpr size_t REGISTER_SET_MAX_TTMPS = 16;

/// @brief Normal SGPRs safe for scratch allocation across supported families.
///
/// @details CDNA exposes 102 ordinary SGPRs per wavefront while RDNA exposes
/// 106. Liveness itself tracks the union, but generic scratch selection must be
/// conservative unless it is made target-ISA-specific.
inline constexpr size_t REGISTER_SET_ALLOCATABLE_SGPRS =
    std::min<size_t>(amdgpu::CdnaIsaBase::MAX_SGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);

/// @brief ISA-independent register-file class.
///
/// @details Each class has its own namespace. For example SGPR 4 and VGPR 4 are
/// different registers, so they must not collide in the same flat bitset. The
/// enum is deliberately small and hardware-oriented; operands that are literals,
/// labels, waitcnt immediates, message IDs, and other non-register values should
/// not produce a RegisterRef.
/// @details SGPR, VGPR, and ACC_VGPR are ordinary indexed register files with a
/// per-index bitset. EXEC, VCC, SCC, M0, FLAT_SCRATCH, and PC are architectural
/// special registers: singletons (there is one EXEC, one SCC, ...), not used for
/// scratch allocation, stored in a compact membership mask. TTMP is a per-index
/// trap-temporary file that this set deliberately does not track — it has
/// neither a bitset nor a mask bit. `is_special_reg_class()` therefore names
/// specifically "held in the special mask" (true for EXEC..PC), not "is it
/// indexed": it is false for TTMP just as for the ordinary classes.
enum class RegClass : uint8_t {
  SGPR,         ///< Scalar general-purpose register, indexed as sN. Ordinary, per-index.
  VGPR,         ///< Vector general-purpose register, indexed as vN. Ordinary, per-index.
  ACC_VGPR,     ///< CDNA accumulator VGPR, indexed as accN. Ordinary, per-index.
  EXEC,         ///< EXEC mask. Special singleton register.
  VCC,          ///< VCC condition mask. Special singleton register.
  SCC,          ///< Scalar condition code bit. Special singleton register.
  M0,           ///< M0 special scalar register. Special singleton register.
  FLAT_SCRATCH, ///< Flat-scratch base pair. Special singleton register.
  TTMP,         ///< Trap-temporary file (ttmpN). Per-index, but untracked here.
  PC,           ///< Program counter/control-flow dep. Special singleton register.
};

/// @brief True if @p cls is an architectural special register (EXEC, VCC, SCC,
/// M0, FLAT_SCRATCH, PC) rather than an ordinary indexed register file
/// (SGPR, VGPR, ACC_VGPR). Special registers are singletons held in the set's
/// special mask; ordinary ones occupy per-index bitsets.
[[nodiscard]] constexpr bool is_special_reg_class(RegClass cls) {
  switch (cls) {
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    return true;
  case RegClass::SGPR:
  case RegClass::VGPR:
  case RegClass::ACC_VGPR:
  case RegClass::TTMP: // per-index, but held in no mask: untracked, not special
    return false;
  }
  return false;
}

/// @brief Widest special-class value that can set a bit in a special mask.
/// @details Derived from `is_special_reg_class` rather than a named enumerator,
/// so appending a RegClass keeps the bound correct without updating callers.
[[nodiscard]] constexpr uint8_t widest_special_reg_class() {
  uint8_t widest = 0;
  for (unsigned v = 0; v <= 0xFF; ++v) {
    if (is_special_reg_class(static_cast<RegClass>(v)))
      widest = static_cast<uint8_t>(v);
  }
  return widest;
}

/// @brief A contiguous register reference within one register file.
///
/// @details `index` is relative to `cls`, not a raw operand encoding value.
/// `width` is measured in 32-bit register lanes. A 64-bit SGPR pair is
/// `{RegClass::SGPR, base, 2}`. The current MR ISA max tracked operand width
/// is 32 lanes (1024-bit MFMA accumulator operands), so uint8_t has ample room.
struct RegisterRef {
  RegClass cls;
  uint16_t index;
  uint8_t width = 1;

  constexpr bool operator==(const RegisterRef &) const = default;
};

/// @brief Storage implementation used by a register set.
enum class RegisterSetWordType { StandardUint64, Avx2M256 };

namespace register_set_detail {
template <RegisterSetWordType wordType> struct WordOps {};

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
};

#if defined(__AVX2__)
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
};
#endif

// Range and set operations shared by scalar and SIMD words.
template <RegisterSetWordType wordType> struct WordArrayOps : WordOps<wordType> {
  using Word = typename WordOps<wordType>::Word;
  using WordOps<wordType>::kWordBits;
  using WordOps<wordType>::word_all_bits;
  using WordOps<wordType>::word_or;
  using WordOps<wordType>::word_and;
  using WordOps<wordType>::word_and_not;
  using WordOps<wordType>::word_none;
  using WordOps<wordType>::word_equal;
  using WordOps<wordType>::word_bit_mask;
  using WordOps<wordType>::word_prefix;
  // Single-bit helpers require bitIndex < kWordBits.
  static void word_set_bit(Word &word, unsigned bitIndex) {
    word = word_or(word, word_bit_mask(bitIndex));
  }
  static void word_reset_bit(Word &word, unsigned bitIndex) {
    word = word_and_not(word, word_bit_mask(bitIndex));
  }
  static bool word_test_bit(Word word, unsigned bitIndex) {
    return !word_none(word_and(word, word_bit_mask(bitIndex)));
  }
  static Word word_range(unsigned begin, unsigned end) {
    return word_and_not(word_prefix(end), word_prefix(begin));
  }

  template <size_t Words> static bool words_equal(const Word (&a)[Words], const Word (&b)[Words]) {
    for (size_t i = 0; i < Words; ++i)
      if (!word_equal(a[i], b[i]))
        return false;
    return true;
  }
  // Visit a nonempty, in-bounds bit range. Stop when an operation returns a
  // value other than kContinue; otherwise return kContinue after the last word.
  // Full words need no range mask. Inline to retain the caller's capacity bounds.
  template <typename WordType, size_t NumWords, typename Operation>
  [[gnu::always_inline]] static inline bool
  visit_range(WordType (&words)[NumWords], size_t startBit, size_t endBit, Operation operation) {
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
    bool partial_word(const Word &word, Word mask) const {
      return !word_none(word_and(word, mask));
    }
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
  static void update_range(Word (&words)[NumWords], size_t startBit, size_t numBits,
                           Update update) {
    if (numBits == 0)
      return;
    if (startBit >= CapacityBits)
      return;
    const size_t endBit = startBit + std::min(numBits, CapacityBits - startBit);
    visit_range(words, startBit, endBit, UpdateWords<Update>{update});
  }
  template <size_t CapacityBits, size_t NumWords>
  static void set_range(Word (&words)[NumWords], size_t startBit, size_t numBits) {
    if (numBits == 1) {
      if (startBit < CapacityBits)
        word_set_bit(words[startBit / kWordBits], startBit % kWordBits);
      return;
    }
    update_range<CapacityBits>(words, startBit, numBits, word_or);
  }
  template <size_t CapacityBits, size_t NumWords>
  static void reset_range(Word (&words)[NumWords], size_t startBit, size_t numBits) {
    if (numBits == 1) {
      if (startBit < CapacityBits)
        word_reset_bit(words[startBit / kWordBits], startBit % kWordBits);
      return;
    }
    update_range<CapacityBits>(words, startBit, numBits, word_and_not);
  }
  template <size_t CapacityBits, size_t NumWords>
  static bool contains_range(const Word (&words)[NumWords], size_t startBit, size_t numBits) {
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
  static bool intersects_range(const Word (&words)[NumWords], size_t startBit, size_t numBits) {
    if (numBits == 1)
      return startBit < CapacityBits &&
             word_test_bit(words[startBit / kWordBits], startBit % kWordBits);
    if (numBits == 0 || startBit >= CapacityBits)
      return false;
    const size_t endBit = startBit + std::min(numBits, CapacityBits - startBit);

    return visit_range(words, startBit, endBit, IntersectsWords{});
  }
  template <size_t NumWords, typename Operation>
  static void combine(Word (&lhs)[NumWords], const Word (&rhs)[NumWords], Operation operation) {
    for (size_t word = 0; word < NumWords; ++word)
      lhs[word] = operation(lhs[word], rhs[word]);
  }
  template <size_t NumWords> static bool bits_none(const Word (&words)[NumWords]) {
    for (const Word word : words)
      if (!word_none(word))
        return false;
    return true;
  }
  template <size_t NumWords>
  static bool intersects_bits(const Word (&lhs)[NumWords], const Word (&rhs)[NumWords]) {
    for (size_t word = 0; word < NumWords; ++word)
      if (!word_none(word_and(lhs[word], rhs[word])))
        return true;
    return false;
  }
};
} // namespace register_set_detail

/// @brief Per-class register set used for def/use and liveness dataflow.
///
/// @details A RegisterSet can represent an instruction's use set, def set,
/// basic-block live-in/live-out set, or live-before/live-after set. It stores
/// the ordinary indexed classes (SGPR, VGPR, AccVGPR) as per-index bitsets and
/// the architectural special registers (EXEC, VCC, ...) as a singleton
/// membership mask. Set operations are member-wise, so the ordinary classes
/// and the special mask all stay disjoint.
///
/// @details Special classes are singleton resources: `expand`, `erase`, and
/// `contains` ignore a special `RegisterRef`'s index/width and act on the
/// class as a whole, and full iteration emits a canonical `{cls, 0, 1}` ref
/// per present special class. The general APIs (`size`, `none`, `for_each`)
/// describe the full set; consumers that reason only about allocatable/indexed
/// registers use the ordinary views (`ordinary_size`, `for_each_ordinary`,
/// `ordinary_only`) so singleton special state never looks spillable or indexed.
/// @tparam wordType The storage implementation. Scalar instantiations remain
/// available in AVX2 builds for testing.
template <RegisterSetWordType wordType> class RegisterSetT {
  using Ops = register_set_detail::WordArrayOps<wordType>;
  using Word = typename Ops::Word;

public:
  using WordType = Word;
  static constexpr RegisterSetWordType kWordType = wordType;

  /// @brief Add `ref`. For ordinary classes this marks every 32-bit lane it
  /// covers; for special classes it marks the singleton (index/width ignored).
  // Keep constant register classes and widths visible in iteration callbacks.
  [[gnu::always_inline]] inline void expand(RegisterRef ref);

  /// @brief Remove `ref`. For ordinary classes this clears every lane it
  /// covers; for special classes it clears the singleton (index/width ignored).
  void erase(RegisterRef ref);

  /// @brief Remove every register in one class (ordinary bitset or special bit).
  void clear_class(RegClass cls);

  /// @brief Return true if `ref` is present. For ordinary classes every covered
  /// lane must be present; for special classes only membership is checked.
  // Inline the class/width dispatch and expose the read-only contract so
  // iteration can omit mutation checks around callbacks using contains().
  [[nodiscard, gnu::pure, gnu::always_inline]] inline bool contains(RegisterRef ref) const;

  /// @brief Return true if any lane covered by `ref` is present.
  ///
  /// @details The any-of counterpart to contains(). Register allocation asks
  /// this to reject a candidate tuple: a run is usable only when none of its
  /// lanes is in the unavailable set. Classes RegisterSetT does not track answer
  /// false, matching contains().
  [[nodiscard]] bool intersects(RegisterRef ref) const;

  /// @brief Return true when neither the ordinary bitsets nor the special mask
  /// hold any member.
  [[nodiscard]] bool none() const;

  /// @brief Total number of members: ordinary single-lane registers plus
  /// distinct special singletons.
  [[nodiscard]] size_t size() const;

  /// @brief Number of ordinary single-lane registers only (SGPR + VGPR +
  /// AccVGPR), excluding special singletons. Use this for scratch/slot sizing.
  [[nodiscard]] size_t ordinary_size() const;

  /// @brief True if any special singleton is present.
  [[nodiscard]] bool has_specials() const { return special_regs_ != 0; }

  /// @brief Return true if any member is present in both sets (ordinary or special).
  [[nodiscard]] bool intersects(const RegisterSetT &rhs) const;

  RegisterSetT &operator|=(const RegisterSetT &rhs);
  RegisterSetT &operator&=(const RegisterSetT &rhs);
  RegisterSetT &operator-=(const RegisterSetT &rhs);

  friend RegisterSetT operator|(RegisterSetT lhs, const RegisterSetT &rhs) {
    lhs |= rhs;
    return lhs;
  }
  friend RegisterSetT operator&(RegisterSetT lhs, const RegisterSetT &rhs) {
    lhs &= rhs;
    return lhs;
  }
  friend RegisterSetT operator-(RegisterSetT lhs, const RegisterSetT &rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend bool operator==(const RegisterSetT &a, const RegisterSetT &b) {
    return Ops::words_equal(a.sgprs_, b.sgprs_) && Ops::words_equal(a.vgprs_, b.vgprs_) &&
           Ops::words_equal(a.acc_vgprs_, b.acc_vgprs_) && a.special_regs_ == b.special_regs_;
  }

  /// @brief Return a copy holding only the ordinary members; special singletons
  /// are dropped. This is the projection scratch/liveness/spill consumers use.
  [[nodiscard]] RegisterSetT ordinary_only() const {
    RegisterSetT copy = *this;
    copy.special_regs_ = 0;
    return copy;
  }

  /// @brief Invoke @p f with each member of the full set.
  ///
  /// @details Visits ordinary SGPRs, VGPRs, then AccVGPRs in ascending index
  /// order (each as a @c width=1 RegisterRef), then each present special class
  /// in ascending `RegClass` order as a canonical `{cls, 0, 1}` ref.
  template <typename F> void for_each(F &&f) const {
    for_each_ordinary(f);
    for_each_special([&](RegClass cls) { f(RegisterRef{cls, 0, 1}); });
  }

  /// @brief Invoke @p f with each ordinary single-lane register (SGPR, VGPR,
  /// AccVGPR) in ascending index order. Special singletons are not visited.
  template <typename F> void for_each_ordinary(F &&f) const {
    for_each_bits<RegClass::SGPR>(sgprs_, f);
    for_each_bits<RegClass::VGPR>(vgprs_, f);
    for_each_bits<RegClass::ACC_VGPR>(acc_vgprs_, f);
  }

  /// @brief Invoke @p f with each present special class, in ascending
  /// `RegClass` value order.
  template <typename F> void for_each_special(F &&f) const {
    uint16_t bits = special_regs_;
    while (bits != 0) {
      const auto i = static_cast<uint8_t>(std::countr_zero(bits));
      f(static_cast<RegClass>(i));
      bits &= static_cast<uint16_t>(bits - 1);
    }
  }

private:
  /// @brief Mask bit for a special class. Only valid for special classes.
  static constexpr uint16_t special_bit(RegClass cls) {
    return static_cast<uint16_t>(1u << static_cast<uint8_t>(cls));
  }

  // The special mask indexes bits by raw RegClass value, so every special class
  // must fit in `special_regs_`. If a special class ever grows past the mask
  // width, widen it (and the shift base in `special_bit`) rather than
  // truncating membership.
  static_assert(widest_special_reg_class() < 16,
                "special_regs_ must hold a bit for every special RegClass");

  // SIMD words support set algebra and lane-occupancy scans on AVX2;
  // other targets use scalar words. expand() clips to the class capacity, so
  // the unused tail bits stay zero under every set operation.
  template <size_t Capacity>
  using RegisterBits = Word[(Capacity + Ops::kWordBits - 1) / Ops::kWordBits];

  // Keep construction and the callback together so the class stays constant.
  // Clang needs a call-site hint for larger callbacks, such as string formatting.
  // Their callees still follow the compiler's normal inlining decisions.
  template <RegClass cls, typename F>
  [[gnu::always_inline]] static inline void emit_register(F &f, uint16_t index) {
#if defined(__clang__)
    [[clang::always_inline]]
#endif
    f(RegisterRef{cls, index, 1});
  }

  // Keep the class constant even when this helper is not inlined.
  template <RegClass cls, size_t Words, typename F>
  static void for_each_bits(const Word (&words)[Words], F &f) {
#if defined(__AVX2__)
    if constexpr (wordType == RegisterSetWordType::Avx2M256) {
      for (size_t word = 0; word < Words; ++word) {
        unsigned lanes = Ops::nonempty_lanes(words[word]);
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
              emit_register<cls>(f, static_cast<uint16_t>(word * 256 + lane * 64 + cursor));
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
            emit_register<cls>(f, static_cast<uint16_t>(word * 256 + lane * 64 + bit));
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
          lanes = Ops::nonempty_lanes(words[word]) & (~1u << lane);
        }
      }
    } else
#endif
    {
      for (size_t word = 0; word < Words; ++word) {
        Word members = words[word];
        Word remaining = members;
        while (remaining) {
          unsigned bit = std::countr_zero(remaining);
          emit_register<cls>(f, static_cast<uint16_t>(word * Ops::kWordBits + bit));
          Word updated = words[word];
          if (updated == members)
            remaining &= remaining - 1;
          else
            remaining =
                bit == Ops::kWordBits - 1 ? 0 : updated & (Ops::word_all_bits() << (bit + 1));
          members = updated;
        }
      }
    }
  }

  RegisterBits<REGISTER_SET_MAX_SGPRS> sgprs_{};
  RegisterBits<REGISTER_SET_MAX_VGPRS> vgprs_{};
  RegisterBits<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_{};

  /// @brief Bit `static_cast<uint8_t>(cls)` set iff special class `cls` is
  /// present. Only special-class bits are ever set (see `expand`).
  uint16_t special_regs_ = 0;
};

template <RegisterSetWordType wordType> void RegisterSetT<wordType>::expand(RegisterRef ref) {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    Ops::template set_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
    break;
  case RegClass::VGPR:
    Ops::template set_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
    break;
  case RegClass::ACC_VGPR:
    Ops::template set_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
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

template <RegisterSetWordType wordType> void RegisterSetT<wordType>::erase(RegisterRef ref) {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    Ops::template reset_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
    break;
  case RegClass::VGPR:
    Ops::template reset_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
    break;
  case RegClass::ACC_VGPR:
    Ops::template reset_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
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

template <RegisterSetWordType wordType> void RegisterSetT<wordType>::clear_class(RegClass cls) {
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

template <RegisterSetWordType wordType>
bool RegisterSetT<wordType>::contains(RegisterRef ref) const {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return Ops::template contains_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
  case RegClass::VGPR:
    return Ops::template contains_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
  case RegClass::ACC_VGPR:
    return Ops::template contains_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, numBits);
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

template <RegisterSetWordType wordType> bool RegisterSetT<wordType>::none() const {
  return Ops::bits_none(sgprs_) && Ops::bits_none(vgprs_) && Ops::bits_none(acc_vgprs_) &&
         special_regs_ == 0;
}

template <RegisterSetWordType wordType> size_t RegisterSetT<wordType>::size() const {
  return ordinary_size() + static_cast<size_t>(std::popcount(special_regs_));
}

template <RegisterSetWordType wordType> size_t RegisterSetT<wordType>::ordinary_size() const {
  return Ops::count_bits(sgprs_) + Ops::count_bits(vgprs_) + Ops::count_bits(acc_vgprs_);
}

template <RegisterSetWordType wordType>
bool RegisterSetT<wordType>::intersects(RegisterRef ref) const {
  const size_t numBits = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return Ops::template intersects_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, numBits);
  case RegClass::VGPR:
    return Ops::template intersects_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, numBits);
  case RegClass::ACC_VGPR:
    return Ops::template intersects_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index,
                                                                      numBits);
  default:
    return false;
  }
}

template <RegisterSetWordType wordType>
bool RegisterSetT<wordType>::intersects(const RegisterSetT &rhs) const {
  return Ops::intersects_bits(sgprs_, rhs.sgprs_) || Ops::intersects_bits(vgprs_, rhs.vgprs_) ||
         Ops::intersects_bits(acc_vgprs_, rhs.acc_vgprs_) ||
         (special_regs_ & rhs.special_regs_) != 0;
}

template <RegisterSetWordType wordType>
RegisterSetT<wordType> &RegisterSetT<wordType>::operator|=(const RegisterSetT &rhs) {
  Ops::combine(sgprs_, rhs.sgprs_, Ops::word_or);
  Ops::combine(vgprs_, rhs.vgprs_, Ops::word_or);
  Ops::combine(acc_vgprs_, rhs.acc_vgprs_, Ops::word_or);
  special_regs_ |= rhs.special_regs_;
  return *this;
}

template <RegisterSetWordType wordType>
RegisterSetT<wordType> &RegisterSetT<wordType>::operator&=(const RegisterSetT &rhs) {
  Ops::combine(sgprs_, rhs.sgprs_, Ops::word_and);
  Ops::combine(vgprs_, rhs.vgprs_, Ops::word_and);
  Ops::combine(acc_vgprs_, rhs.acc_vgprs_, Ops::word_and);
  special_regs_ &= rhs.special_regs_;
  return *this;
}

template <RegisterSetWordType wordType>
RegisterSetT<wordType> &RegisterSetT<wordType>::operator-=(const RegisterSetT &rhs) {
  Ops::combine(sgprs_, rhs.sgprs_, Ops::word_and_not);
  Ops::combine(vgprs_, rhs.vgprs_, Ops::word_and_not);
  Ops::combine(acc_vgprs_, rhs.acc_vgprs_, Ops::word_and_not);
  special_regs_ &= static_cast<uint16_t>(~rhs.special_regs_);
  return *this;
}

#if defined(__AVX2__)
using RegisterSet = RegisterSetT<RegisterSetWordType::Avx2M256>;
#else
using RegisterSet = RegisterSetT<RegisterSetWordType::StandardUint64>;
#endif

} // namespace rocjitsu
