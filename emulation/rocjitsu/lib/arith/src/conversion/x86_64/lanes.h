// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Intrinsic-backed unsigned lanes for the shared raw-bit conversion model.
// Masks use all-one/all-zero lanes. This is conversion-local, not a general SIMD API.
#pragma once

#include "common/isa.h"

#include <immintrin.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd {

// Evaluate the conversion model on 32- or 64-bit lanes under this TU's ISA.
template <int Bits> class Lanes {
public:
  static_assert(Bits == 32 || Bits == 64);
  using value_type = std::conditional_t<Bits == 32, uint32_t, uint64_t>;
#if defined(__AVX512F__)
  using Register = __m512i;
#else
  using Register = __m256i;
#endif
  Register bits;

  Lanes() = default;
  // Broadcast an unsigned scalar to every lane.
  Lanes(value_type value) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      bits = _mm512_set1_epi32(value);
    } else {
      bits = _mm512_set1_epi64(value);
    }
#else
    if constexpr (Bits == 32) {
      bits = _mm256_set1_epi32(value);
    } else {
      bits = _mm256_set1_epi64x(value);
    }
#endif
  }
  // Wrap raw intrinsic lanes without numerical conversion.
  explicit Lanes(Register value) : bits(value) {}

  // Load a full unaligned block of raw encodings.
  explicit Lanes(const value_type *source) {
#if defined(__AVX512F__)
    bits = _mm512_loadu_si512(source);
#else
    bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
#endif
  }

  // Number of independent values in one register.
  static constexpr int size() { return sizeof(Register) * 8 / Bits; }

  // Load 32-bit words, widening each one to a 64-bit lane.
  static Lanes load_words(const uint32_t *source) {
    static_assert(Bits == 64);
#if defined(__AVX512F__)
    return Lanes(
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(source))));
#else
    return Lanes(_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source))));
#endif
  }

  // Write each 64-bit lane's low word to consecutive 32-bit destinations.
  void store_words(uint32_t *destination) const {
    static_assert(Bits == 64);
#if defined(__AVX512F__)
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination), _mm512_cvtepi64_epi32(bits));
#else
    auto packed = _mm256_permutevar8x32_epi32(bits, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(destination), _mm256_castsi256_si128(packed));
#endif
  }

  // Store only EXEC-selected lanes; pointers need not be aligned.
  void store(uint32_t *destination, uint32_t exec_mask) const {
    static_assert(Bits == 32);
#if defined(__AVX512F__)
    _mm512_mask_storeu_epi32(destination, __mmask16(exec_mask), bits);
#else
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(destination), active, bits);
#endif
  }

  // Operate independently on unsigned lanes.
  friend Lanes operator+(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_add_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm512_add_epi64(a.bits, b.bits));
    }
#else
    if constexpr (Bits == 32) {
      return Lanes(_mm256_add_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm256_add_epi64(a.bits, b.bits));
    }
#endif
  }

  // Operate independently on unsigned lanes.
  friend Lanes operator-(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_sub_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm512_sub_epi64(a.bits, b.bits));
    }
#else
    if constexpr (Bits == 32) {
      return Lanes(_mm256_sub_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm256_sub_epi64(a.bits, b.bits));
    }
#endif
  }

  // Lane-wise unsigned arithmetic; shifts at or above the lane width produce zero.
  friend Lanes operator<<(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_sllv_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm512_sllv_epi64(a.bits, b.bits));
    }
#else
    if constexpr (Bits == 32) {
      return Lanes(_mm256_sllv_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm256_sllv_epi64(a.bits, b.bits));
    }
#endif
  }

  // Operate independently on unsigned lanes.
  friend Lanes operator>>(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_srlv_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm512_srlv_epi64(a.bits, b.bits));
    }
#else
    if constexpr (Bits == 32) {
      return Lanes(_mm256_srlv_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm256_srlv_epi64(a.bits, b.bits));
    }
#endif
  }

  // Combine corresponding encoding bits.
  friend Lanes operator&(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    return Lanes(_mm512_and_si512(a.bits, b.bits));
#else
    return Lanes(_mm256_and_si256(a.bits, b.bits));
#endif
  }

  // Combine corresponding encoding bits.
  friend Lanes operator|(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    return Lanes(_mm512_or_si512(a.bits, b.bits));
#else
    return Lanes(_mm256_or_si256(a.bits, b.bits));
#endif
  }

  // Combine corresponding encoding bits.
  friend Lanes operator^(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    return Lanes(_mm512_xor_si512(a.bits, b.bits));
#else
    return Lanes(_mm256_xor_si256(a.bits, b.bits));
#endif
  }

  // Invert encoding bits, also complementing a comparison mask.
  friend Lanes operator~(Lanes a) { return a ^ Lanes(~value_type(0)); }

  // Update bits accumulated by the shared model's bit-width and decoding steps.
  Lanes &operator|=(Lanes b) { return *this = *this | b; }

  // Comparisons return all-one/all-zero lanes, with unsigned ordering.
  friend Lanes operator==(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a.bits, b.bits), -1));
    } else {
      return Lanes(_mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask(a.bits, b.bits), -1));
    }
#else
    if constexpr (Bits == 32) {
      return Lanes(_mm256_cmpeq_epi32(a.bits, b.bits));
    } else {
      return Lanes(_mm256_cmpeq_epi64(a.bits, b.bits));
    }
#endif
  }

  // Compare unsigned lanes without interpreting a high sign bit as negative.
  friend Lanes operator<(Lanes a, Lanes b) {
#if defined(__AVX512F__)
    if constexpr (Bits == 32) {
      return Lanes(_mm512_maskz_set1_epi32(_mm512_cmplt_epu32_mask(a.bits, b.bits), -1));
    } else {
      return Lanes(_mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(a.bits, b.bits), -1));
    }
#else
    const Lanes sign(value_type(1) << (Bits - 1));
    a = a ^ sign;
    b = b ^ sign;
    if constexpr (Bits == 32) {
      return Lanes(_mm256_cmpgt_epi32(b.bits, a.bits));
    } else {
      return Lanes(_mm256_cmpgt_epi64(b.bits, a.bits));
    }
#endif
  }

  // Derived comparisons retain the all-one/all-zero mask representation.
  friend Lanes operator!=(Lanes a, Lanes b) { return ~(a == b); }
  friend Lanes operator>(Lanes a, Lanes b) { return b < a; }
  friend Lanes operator>=(Lanes a, Lanes b) { return ~(a < b); }

  // Boolean operations consume comparison masks, not arbitrary integer values.
  friend Lanes operator!(Lanes a) { return ~a; }
  friend Lanes operator&&(Lanes a, Lanes b) { return a & b; }
  friend Lanes operator||(Lanes a, Lanes b) { return a | b; }
};

// ADL hook for the model's scalar/SIMD choose: mask lanes are all-one/all-zero.
template <int Bits>
Lanes<Bits> select(Lanes<Bits> mask, Lanes<Bits> when_true, Lanes<Bits> when_false) {
  return (mask & when_true) | (~mask & when_false);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd
