// SPDX-License-Identifier: MIT

// AVX2 operations and dispatch for the shared dense INT8/INT4 WMMA algorithm.
// Processes eight output columns at a time using widened 16-bit pair products;
// register layouts and accumulation stages are defined in integer_wmma.h.

#include "common/simd.h"
#include "wmma/x86_64/integer_wmma.h"

#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m256i;
  static const int width = 8;

  // Broadcast a signed 32-bit value to every lane.
  static V splat(int x) { return _mm256_set1_epi32(x); }

  // Broadcast a 32-bit bit pattern without changing its signed representation.
  static V splat_bits(uint32_t x) {
    int32_t s;
    std::memcpy(&s, &x, sizeof(s));
    return splat(s);
  }

  // Load width consecutive words; x need not be vector-aligned.
  static V load(const uint32_t *x) { return _mm256_loadu_si256(reinterpret_cast<const V *>(x)); }

  // Store width consecutive words; x need not be vector-aligned.
  static void store(uint32_t *x, V v) { _mm256_storeu_si256(reinterpret_cast<V *>(x), v); }

  // Add 32-bit lanes modulo 2^32.
  static V add(V a, V b) { return _mm256_add_epi32(a, b); }

  // Subtract 32-bit lanes modulo 2^32.
  static V sub(V a, V b) { return _mm256_sub_epi32(a, b); }

  // Return the bitwise intersection of a and b.
  static V bit_and(V a, V b) { return _mm256_and_si256(a, b); }

  // Return the bitwise union of a and b.
  static V bit_or(V a, V b) { return _mm256_or_si256(a, b); }

  // Return the bitwise exclusive-or of a and b.
  static V bit_xor(V a, V b) { return _mm256_xor_si256(a, b); }

  // Logically shift each 32-bit lane right by n bits.
  static V shift_right(V a, int n) { return _mm256_srl_epi32(a, _mm_cvtsi32_si128(n)); }

  // Move each lane's low halfword into its high halfword, clearing the low half.
  static V shift_left16(V a) { return _mm256_slli_epi32(a, 16); }

  // Return all-one bits for negative 32-bit lanes, zero otherwise.
  static V sign(V a) { return _mm256_srai_epi32(a, 31); }

  // Add adjacent signed 16-bit products to each 32-bit sum, without saturation.
  static V dot(V sum, V a, V b) { return _mm256_add_epi32(sum, _mm256_madd_epi16(a, b)); }

  // Select yes where the corresponding signed 32-bit condition is negative; no elsewhere.
  static V select_negative(V condition, V yes, V no) {
    return _mm256_blendv_epi8(no, yes, sign(condition));
  }
};

} // namespace

namespace rocjitsu::arith {

// Compute a gfx12 integer tile in Wave32 layout; D may alias A/B/C.
// Requires validated modifiers and (bits, k) of (8,16), (4,16), or (4,32).
void integer_wmma_x86_64_v3(int bits, int k, uint32_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (bits == 8) {
    integer_wmma<8, 16, Ops>(modifiers, d, a, b, c);
  } else if (k == 16) {
    integer_wmma<4, 16, Ops>(modifiers, d, a, b, c);
  } else {
    integer_wmma<4, 32, Ops>(modifiers, d, a, b, c);
  }
}

// Compute a gfx11 integer tile for the selected wave size and factor width.
// Requires validated modifiers; D may overlap source registers.
template <int Bits, int Lanes>
void integer_wmma_replicated_x86_64_v3(uint32_t modifiers, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  integer_wmma<Bits, 16, Ops, Lanes, true>(modifiers, d, a, b, c);
}

template void integer_wmma_replicated_x86_64_v3<4, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<4, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<8, 32>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void integer_wmma_replicated_x86_64_v3<8, 64>(uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
