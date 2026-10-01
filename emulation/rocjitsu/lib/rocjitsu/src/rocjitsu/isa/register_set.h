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

namespace register_set_detail {
#if defined(__AVX2__)
using Word = __m256i;
inline constexpr size_t kWordBits = 256;
inline Word word_all_bits() { return _mm256_set1_epi64x(-1); }
// One occupancy bit per 64-bit lane, scanning all 256 membership bits at once.
inline unsigned nonempty_lanes(Word word) {
  const Word empty = _mm256_cmpeq_epi64(word, _mm256_setzero_si256());
  return ~static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(empty))) & 15u;
}
inline Word word_or(Word a, Word b) { return _mm256_or_si256(a, b); }
inline Word word_and(Word a, Word b) { return _mm256_and_si256(a, b); }
// Computes a & ~b; the intrinsic complements its first operand.
inline Word word_and_not(Word a, Word b) { return _mm256_andnot_si256(b, a); }
inline bool word_none(Word a) { return _mm256_testz_si256(a, a) != 0; }
inline bool word_equal(Word a, Word b) {
  return _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) == -1;
}

// XOR clears the lane-select bits only in the selected lane. Its shift is
// bitIndex % 64; all other shifts are >= 64 and therefore produce zero.
inline Word word_bit_mask(unsigned bitIndex) {
  const Word shiftBits =
      _mm256_xor_si256(_mm256_set1_epi64x(bitIndex), _mm256_setr_epi64x(0, 64, 128, 192));
  return _mm256_sllv_epi64(_mm256_set1_epi64x(1), shiftBits);
}

// A prefix of [0, end), including the empty and full-word cases.
inline Word word_prefix(unsigned end) {
  const Word counts = _mm256_sub_epi64(_mm256_set1_epi64x(end), _mm256_set_epi64x(192, 128, 64, 0));
  const Word positive = _mm256_cmpgt_epi64(counts, _mm256_setzero_si256());
  const Word high = _mm256_sllv_epi64(word_all_bits(), counts);
  return _mm256_andnot_si256(high, positive);
}
inline Word word_count_lanes(Word bits) {
  const Word table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1,
                                      2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const Word low_mask = _mm256_set1_epi8(15);
  const Word low = _mm256_shuffle_epi8(table, _mm256_and_si256(bits, low_mask));
  const Word high =
      _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits, 4), low_mask));
  return _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256());
}
inline size_t sum_count_lanes(Word sums) {
  const __m128i halves =
      _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
  const __m128i total = _mm_add_epi64(halves, _mm_unpackhi_epi64(halves, halves));
  return static_cast<size_t>(_mm_cvtsi128_si64(total));
}
#else
using Word = uint64_t;
inline constexpr size_t kWordBits = 64;
inline Word word_all_bits() { return ~Word{0}; }
inline Word word_or(Word a, Word b) { return a | b; }
inline Word word_and(Word a, Word b) { return a & b; }
inline Word word_and_not(Word a, Word b) { return a & ~b; }
inline bool word_none(Word a) { return a == 0; }
inline bool word_equal(Word a, Word b) { return a == b; }
inline Word word_prefix(unsigned end) {
  return end == kWordBits ? word_all_bits() : (Word{1} << end) - 1;
}
inline size_t word_count(Word bits) { return std::popcount(bits); }
inline Word word_bit_mask(unsigned bitIndex) { return Word{1} << bitIndex; }
#endif
// Single-bit helpers require bitIndex < kWordBits.
inline void word_set_bit(Word &word, unsigned bitIndex) {
  word = word_or(word, word_bit_mask(bitIndex));
}
inline void word_reset_bit(Word &word, unsigned bitIndex) {
  word = word_and_not(word, word_bit_mask(bitIndex));
}
inline bool word_test_bit(Word word, unsigned bitIndex) {
  return !word_none(word_and(word, word_bit_mask(bitIndex)));
}
inline Word word_range(unsigned begin, unsigned end) {
  return word_and_not(word_prefix(end), word_prefix(begin));
}

template <size_t Words> bool words_equal(const Word (&a)[Words], const Word (&b)[Words]) {
  for (size_t i = 0; i < Words; ++i)
    if (!word_equal(a[i], b[i]))
      return false;
  return true;
}
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
class RegisterSet {
public:
  /// @brief Add `ref`. For ordinary classes this marks every 32-bit lane it
  /// covers; for special classes it marks the singleton (index/width ignored).
  void expand(RegisterRef ref);

  /// @brief Remove `ref`. For ordinary classes this clears every lane it
  /// covers; for special classes it clears the singleton (index/width ignored).
  void erase(RegisterRef ref);

  /// @brief Remove every register in one class (ordinary bitset or special bit).
  void clear_class(RegClass cls);

  /// @brief Return true if `ref` is present. For ordinary classes every covered
  /// lane must be present; for special classes only membership is checked.
  // Expose the read-only contract across translation units so inlined
  // iteration can omit mutation checks around callbacks using contains().
  [[nodiscard, gnu::pure]] bool contains(RegisterRef ref) const;

  /// @brief Return true if any lane covered by `ref` is present.
  ///
  /// @details The any-of counterpart to contains(). Register allocation asks
  /// this to reject a candidate tuple: a run is usable only when none of its
  /// lanes is in the unavailable set. Classes RegisterSet does not track answer
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
  [[nodiscard]] bool intersects(const RegisterSet &rhs) const;

  RegisterSet &operator|=(const RegisterSet &rhs);
  RegisterSet &operator&=(const RegisterSet &rhs);
  RegisterSet &operator-=(const RegisterSet &rhs);

  friend RegisterSet operator|(RegisterSet lhs, const RegisterSet &rhs) {
    lhs |= rhs;
    return lhs;
  }
  friend RegisterSet operator&(RegisterSet lhs, const RegisterSet &rhs) {
    lhs &= rhs;
    return lhs;
  }
  friend RegisterSet operator-(RegisterSet lhs, const RegisterSet &rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend bool operator==(const RegisterSet &a, const RegisterSet &b) {
    using register_set_detail::words_equal;
    return words_equal(a.sgprs_, b.sgprs_) && words_equal(a.vgprs_, b.vgprs_) &&
           words_equal(a.acc_vgprs_, b.acc_vgprs_) && a.special_regs_ == b.special_regs_;
  }

  /// @brief Return a copy holding only the ordinary members; special singletons
  /// are dropped. This is the projection scratch/liveness/spill consumers use.
  [[nodiscard]] RegisterSet ordinary_only() const {
    RegisterSet copy = *this;
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
  using RegisterBits = register_set_detail::Word[(Capacity + register_set_detail::kWordBits - 1) /
                                                 register_set_detail::kWordBits];

  // Keep the class constant even when this helper is not inlined. Inline the
  // callback's available callees too: opaque vector/string helpers otherwise
  // obstruct alias analysis and retain costly membership reloads and checks.
  template <RegClass cls, size_t Words, typename F>
  [[gnu::flatten]] static void for_each_bits(const register_set_detail::Word (&words)[Words],
                                             F &f) {
    using namespace register_set_detail;
#if defined(__AVX2__)
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
            f(RegisterRef{cls, static_cast<uint16_t>(word * 256 + lane * 64 + cursor), 1});
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
          f(RegisterRef{cls, static_cast<uint16_t>(word * 256 + lane * 64 + bit), 1});
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
#else
    for (size_t word = 0; word < Words; ++word) {
      Word members = words[word];
      Word remaining = members;
      while (remaining) {
        unsigned bit = std::countr_zero(remaining);
        f(RegisterRef{cls, static_cast<uint16_t>(word * 64 + bit), 1});
        Word updated = words[word];
        if (updated == members)
          remaining &= remaining - 1;
        else
          remaining = bit == 63 ? 0 : updated & (~Word{0} << (bit + 1));
        members = updated;
      }
    }
#endif
  }

  RegisterBits<REGISTER_SET_MAX_SGPRS> sgprs_{};
  RegisterBits<REGISTER_SET_MAX_VGPRS> vgprs_{};
  RegisterBits<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_{};

  /// @brief Bit `static_cast<uint8_t>(cls)` set iff special class `cls` is
  /// present. Only special-class bits are ever set (see `expand`).
  uint16_t special_regs_ = 0;
};

} // namespace rocjitsu
