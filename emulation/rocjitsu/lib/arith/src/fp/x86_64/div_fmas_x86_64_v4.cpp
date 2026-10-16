// SPDX-License-Identifier: MIT

// AVX-512 division FMA with conditional scaling and guest rounding/status handling.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/div_fmas.h"
#include "fp/x86_64/div_fmas_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m512i;
  using Mask = __mmask8;
  static constexpr int lanes = 8;

  // Return whether any lane satisfies the supplied predicate.
  static bool any(Mask m) { return m != 0; }

  // Broadcast the supplied bit pattern to each vector lane.
  static V set(uint64_t a) { return _mm512_set1_epi64(int64_t(a)); }

  // Add corresponding integer lanes with unsigned wraparound.
  static V add(V a, V b) { return _mm512_add_epi64(a, b); }

  // Subtract corresponding integer lanes with unsigned wraparound.
  static V sub(V a, V b) { return _mm512_sub_epi64(a, b); }

  // Return the bitwise intersection of the vector operands.
  static V band(V a, V b) { return _mm512_and_si512(a, b); }

  // Return the bitwise union of the vector operands.
  static V bor(V a, V b) { return _mm512_or_si512(a, b); }

  // Return the bitwise exclusive-or of the vector operands.
  static V bxor(V a, V b) { return _mm512_xor_si512(a, b); }

  // Shift each lane left by the supplied count.
  static V shl(V a, V b) { return _mm512_sllv_epi64(a, b); }

  // Shift each unsigned lane right by the supplied count.
  static V shr(V a, V b) { return _mm512_srlv_epi64(a, b); }

  // Multiply the low unsigned 32-bit factors of each 64-bit lane, returning full 64-bit
  // products.
  static V mul32(V a, V b) { return _mm512_mul_epu32(a, b); }

  // Return the lane mask for equal operands.
  static Mask eq(V a, V b) { return _mm512_cmpeq_epi64_mask(a, b); }

  // Return the lane mask for signed a > b.
  static Mask gt(V a, V b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT); }

  // Return the lane mask for unsigned a > b.
  static Mask ugt(V a, V b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_GT); }

  // Intersect lane predicates.
  static Mask both(Mask a, Mask b) { return a & b; }

  // Unite lane predicates.
  static Mask either(Mask a, Mask b) { return a | b; }

  // Complement a lane predicate.
  static Mask inverse(Mask a) { return Mask(~a); }

  // Select values lane by lane according to the supplied predicate.
  static V select(Mask m, V yes, V no) { return _mm512_mask_blend_epi64(m, no, yes); }

  // Expand EXEC bits into the vector lane-predicate representation.
  static Mask lane_mask(uint32_t exec_mask) { return __mmask8(exec_mask); }

  // Load consecutive unsigned 32-bit words and widen them into 64-bit lanes.
  static V load_words(const uint32_t *p) {
    return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
  }

  // Store each 64-bit lane's low word to consecutive 32-bit destinations.
  static void store_words(uint32_t *p, V value) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), _mm512_cvtepi64_epi32(value));
  }

  // Copy staged words to destination lanes selected by exec_mask.
  static void masked_words(uint32_t *d, const uint32_t *p, uint32_t exec_mask) {
    _mm256_mask_storeu_epi32(d, __mmask8(exec_mask),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
  }
};

} // namespace

namespace rocjitsu::arith {

// Evaluate division FMA with conditional scaling. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
void div_fmas_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t condition, bool flush_output,
                        uint32_t *state) {
  division_vector_run<Width, Ops>(exec_mask, mode, d, a, b, c, condition, flush_output, state);
}

template void div_fmas_x86_64_v4<32>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t,
                                     bool, uint32_t *);
template void div_fmas_x86_64_v4<64>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t,
                                     bool, uint32_t *);

} // namespace rocjitsu::arith
