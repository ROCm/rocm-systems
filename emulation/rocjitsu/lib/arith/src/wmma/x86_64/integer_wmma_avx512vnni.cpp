// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 VNNI operations and dispatch for dense INT8/INT4 WMMA.
// Uses byte quartets and saturating word dots for the staged integer forms.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/integer_wmma.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Each 128-bit VPSHUFB lane holds four packed words.
__m512i word_shuffle(int indices) {
  return _mm512_add_epi8(_mm512_set1_epi32(indices),
                         _mm512_set4_epi32(0x0c0c0c0c, 0x08080808, 0x04040404, 0));
}

// Expand the selected adjacent byte pair. With signed factors bias is 128.
__m512i byte_pair(__m512i words, int k, __m512i bias) {
  auto pair = _mm512_shuffle_epi8(words, word_shuffle(k & 2 ? 0x80038002 : 0x80018000));
  return _mm512_sub_epi16(_mm512_xor_si512(pair, bias), bias);
}

// Pack four nibbles from one hardware stage into the bytes of a dot operand.
template <bool Adjacent> __m512i packed_nibbles(__m512i words, int k) {
  // If the whole packed word belongs to one stage, multiply all even nibbles
  // and then all odd nibbles. Each is already in its own byte.
  if constexpr (!Adjacent) {
    return _mm512_and_si512(k & 4 ? _mm512_srli_epi16(words, 4) : words, _mm512_set1_epi8(15));
  }
  auto bytes = _mm512_shuffle_epi8(words, word_shuffle(k & 4 ? 0x03030202 : 0x01010000));
  auto nibbles =
      _mm512_mask_blend_epi8(UINT64_C(0xaaaaaaaaaaaaaaaa), bytes, _mm512_srli_epi16(bytes, 4));
  return _mm512_and_si512(nibbles, _mm512_set1_epi8(15));
}

// VPDPBUSD takes unsigned bytes on the left and signed bytes on the right.
// Swap puts unsigned B on the left. Unbiased IU4 forms keep both operands
// directly in range (unsigned nibbles fit in signed bytes); otherwise bias only
// A so one column sum of B corrects every output row. Gfx11 clamped IU8 instead
// saturates each word dot.
template <int Bits, bool Rdna3, bool Swap, bool Clamp, bool Direct = false> class Products {
  static constexpr bool Pair = Bits == 8 && Rdna3 && Clamp;
  static constexpr bool Adjacent = Rdna3 && Clamp;
  static constexpr bool DirectSat = Direct && Clamp && Rdna3 && Bits == 4;

public:
  using V = __m512i;
  using Accumulator = V;
  static constexpr int width = 16;
  static constexpr int products = Pair ? 2 : 4;
  static constexpr bool biased = !Pair && !Direct;
  static constexpr bool staged_clamp = Clamp;
  static constexpr bool early_store = false;

  explicit Products(uint32_t flags) {
    bool signed_a = flags & RJ_ARITH_WMMA_SIGNED_A;
    bool signed_b = flags & RJ_ARITH_WMMA_SIGNED_B;
    int offset = Direct ? (signed_a ? 8 : 0) : (signed_a != Swap ? 1 << (Bits - 1) : 0);
    bias_a = Pair ? _mm512_set1_epi16(signed_a ? 128 : 0) : _mm512_set1_epi8(offset);
    bias_b = Pair ? _mm512_set1_epi16(signed_b ? 128 : 0) : _mm512_set1_epi8(signed_b ? 8 : 0);
  }

  // Unaligned register transfers and single packed-factor broadcasts.
  static V load(const uint32_t *p) { return _mm512_loadu_si512(p); }
  static void store(uint32_t *p, V v) { _mm512_storeu_si512(p, v); }
  static V broadcast(uint32_t word) { return _mm512_set1_epi32(word); }

  // Convert A into the unsigned or signed byte operand selected by Swap.
  V unpack_a(V word, int k) const {
    if constexpr (Pair) {
      return byte_pair(word, k, bias_a);
    }
    if constexpr (Bits == 4) {
      word = packed_nibbles<Adjacent>(word, k);
    }
    if constexpr (Direct) {
      if constexpr (Bits == 4 && Swap) {
        return _mm512_sub_epi8(_mm512_xor_si512(word, bias_a), bias_a);
      }
      return word;
    }
    return _mm512_xor_si512(word, bias_a);
  }

  // Extend signed nibbles before the byte dot. IU8 keeps B's input bytes.
  V unpack_b(V word, int k) const {
    if constexpr (Pair) {
      return byte_pair(word, k, bias_b);
    }
    if constexpr (Bits == 4) {
      word = packed_nibbles<Adjacent>(word, k);
      if constexpr (Direct && Swap) {
        return word;
      }
      return _mm512_sub_epi8(_mm512_xor_si512(word, bias_b), bias_b);
    }
    return word;
  }

  // Apply A's byte offset to each B column using the same signedness as its dot.
  // In the swapped IU8 dot the byte offset 0x80 represents -128.
  V sum_b(V sum, V b) const {
    return Swap ? _mm512_dpbusd_epi32(sum, b, bias_a) : _mm512_dpbusd_epi32(sum, bias_a, b);
  }
  static V bias_correction(V sum) { return _mm512_sub_epi32(V{}, sum); }

  static V accumulator(V c) { return c; }
  static V result(V acc) { return acc; }

  // A clamped byte stage forms its complete signed contribution before adding C.
  static V start_stage(V acc, V bias) {
    if constexpr (Pair || DirectSat) {
      return acc;
    }
    if constexpr (Direct) {
      return Clamp ? V{} : acc;
    }
    return Clamp ? bias : _mm512_add_epi32(acc, bias);
  }
  static V product(V sum, V a, V b) {
    if constexpr (Pair) {
      return _mm512_dpwssds_epi32(sum, a, b);
    }
    if constexpr (DirectSat) {
      return Swap ? _mm512_dpbusds_epi32(sum, b, a) : _mm512_dpbusds_epi32(sum, a, b);
    }
    return Swap ? _mm512_dpbusd_epi32(sum, b, a) : _mm512_dpbusd_epi32(sum, a, b);
  }

  // IU4 stage sums fit in signed 16 bits. The widest IU8 stage fits in
  // signed 24 bits, so sum = unsigned byte 0 + 256 * signed bytes 1..2.
  // One saturating word dot applies the full stage at the hardware boundary.
  static void finish_stage(V &acc, V sum) {
    if constexpr (!Clamp || Pair || DirectSat) {
      acc = sum;
    } else if constexpr (Bits == 4) {
      acc = _mm512_dpwssds_epi32(acc, sum, _mm512_set1_epi32(1));
    } else {
      auto words = _mm512_shuffle_epi8(sum, word_shuffle(0x02018000));
      acc = _mm512_dpwssds_epi32(acc, words, _mm512_set1_epi32(0x01000001));
    }
  }

private:
  V bias_a, bias_b;
};

// Evaluate a validated dense integer instruction in its physical wave layout.
template <int Bits, int K, int Lanes = 32, bool Rdna3 = false>
void dense_integer(uint32_t flags, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *const *b, const uint32_t *const *c) {
  const auto run = [&](auto dot) {
    rocjitsu::arith::integer_wmma_tile<Bits, K, Lanes, Rdna3, 4>(dot, d, a, b, c);
  };
  const auto dispatch = [&]<bool Clamp>() {
    if constexpr (Rdna3 && Bits == 8 && Clamp) {
      run(Products<Bits, Rdna3, false, true>(flags));
    } else if constexpr (Bits == 4) {
      // Unsigned nibbles [0..15] fit in both unsigned and signed bytes. When B
      // is unsigned, swap operands so A sits in VPDPBUSD's broadcastable third
      // operand with no bias correction.
      if (!(flags & RJ_ARITH_WMMA_SIGNED_B)) {
        run(Products<Bits, Rdna3, true, Clamp, true>(flags));
      } else {
        run(Products<Bits, Rdna3, false, Clamp>(flags));
      }
    } else if (!(flags & RJ_ARITH_WMMA_SIGNED_B)) {
      run(Products<Bits, Rdna3, true, Clamp>(flags));
    } else {
      run(Products<Bits, Rdna3, false, Clamp>(flags));
    }
  };
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  // The x86-64-v4 backend handles wrapping on gfx12 K=32 IU4.
  constexpr bool clamp_only = Bits == 4 && K == 32;
#else
  constexpr bool clamp_only = false;
#endif
  if (clamp_only || (flags & RJ_ARITH_WMMA_CLAMP)) {
    dispatch.template operator()<true>();
  } else {
    dispatch.template operator()<false>();
  }
}

} // namespace

namespace rocjitsu::arith {

// Compute a gfx12 integer tile in Wave32 layout; D may alias A/B/C.
// Requires validated modifiers and a supported compile-time shape.
template <int Bits, int K>
void integer_wmma_avx512vnni(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *c) {
  static_assert((Bits == 8 && K == 16) || (Bits == 4 && (K == 16 || K == 32)));
  dense_integer<Bits, K>(modifiers, d, a, b, c);
}

template void integer_wmma_avx512vnni<8, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_avx512vnni<4, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_avx512vnni<4, 32>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);

// Compute a gfx11 integer tile for the selected wave size and factor width.
// Requires validated modifiers; D may overlap source registers.
template <int Bits, int Lanes>
void integer_wmma_replicated_avx512vnni(uint32_t modifiers, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  dense_integer<Bits, 16, Lanes, true>(modifiers, d, a, b, c);
}

template void integer_wmma_replicated_avx512vnni<4, 32>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void integer_wmma_replicated_avx512vnni<4, 64>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void integer_wmma_replicated_avx512vnni<8, 32>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void integer_wmma_replicated_avx512vnni<8, 64>(uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);

} // namespace rocjitsu::arith
