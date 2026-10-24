// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 VNNI operations and dispatch for dense INT8/INT4 WMMA.
// Processes sixteen output columns with non-saturating 16-bit pair dot products;
// integer_wmma.h supplies layouts and the instruction's saturation stages.

#include "common/simd.h"
#include "wmma/x86_64/integer_wmma.h"

#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace {

class Ops {
public:
  using V = __m512i;
  static const int width = 16;

  // Broadcast a signed 32-bit value to every lane.
  static V splat(int x) { return _mm512_set1_epi32(x); }

  // Broadcast a 32-bit bit pattern without changing its signed representation.
  static V splat_bits(uint32_t x) {
    int32_t s;
    std::memcpy(&s, &x, sizeof(s));
    return splat(s);
  }

  // Load width consecutive words; x need not be vector-aligned.
  static V load(const uint32_t *x) { return _mm512_loadu_si512(reinterpret_cast<const V *>(x)); }

  // Store width consecutive words; x need not be vector-aligned.
  static void store(uint32_t *x, V v) { _mm512_storeu_si512(reinterpret_cast<V *>(x), v); }

  // Add 32-bit lanes modulo 2^32.
  static V add(V a, V b) { return _mm512_add_epi32(a, b); }

  // Subtract 32-bit lanes modulo 2^32.
  static V sub(V a, V b) { return _mm512_sub_epi32(a, b); }

  // Return the bitwise intersection of a and b.
  static V bit_and(V a, V b) { return _mm512_and_si512(a, b); }

  // Return the bitwise union of a and b.
  static V bit_or(V a, V b) { return _mm512_or_si512(a, b); }

  // Return the bitwise exclusive-or of a and b.
  static V bit_xor(V a, V b) { return _mm512_xor_si512(a, b); }

  // Logically shift each 32-bit lane right by n bits.
  static V shift_right(V a, int n) { return _mm512_srl_epi32(a, _mm_cvtsi32_si128(n)); }

  // Move each lane's low halfword into its high halfword, clearing the low half.
  static V shift_left16(V a) { return _mm512_slli_epi32(a, 16); }

  // Return all-one bits for negative 32-bit lanes, zero otherwise.
  static V sign(V a) { return _mm512_srai_epi32(a, 31); }

  // Add adjacent signed 16-bit products to each 32-bit sum, without saturation.
  static V dot(V sum, V a, V b) { return _mm512_dpwssd_epi32(sum, a, b); }

  // Select yes where the corresponding signed 32-bit mask is negative; no elsewhere.
  static V select_negative(V mask, V yes, V no) {
    return _mm512_mask_mov_epi32(no, _mm512_cmp_epi32_mask(mask, splat(0), _MM_CMPINT_LT), yes);
  }
};

} // namespace

namespace rocjitsu::arith {

// Compute a gfx12 integer tile in Wave32 layout; D may alias A/B/C.
// Requires validated modifiers and a supported compile-time shape.
template <int Bits, int K>
void integer_wmma_avx512vnni(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *c) {
  static_assert((Bits == 8 && K == 16) || (Bits == 4 && (K == 16 || K == 32)));
  integer_wmma<Bits, K, Ops>(modifiers, d, a, b, c);
}

template void integer_wmma_avx512vnni<8, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_avx512vnni<4, 16>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);
template void integer_wmma_avx512vnni<4, 32>(uint32_t, uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
