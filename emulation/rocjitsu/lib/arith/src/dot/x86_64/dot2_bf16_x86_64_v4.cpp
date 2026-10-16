// SPDX-License-Identifier: MIT

// AVX-512 loose BF16 DOT2 using pair-dot instructions with exceptional-input handling.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Select values lane by lane according to the supplied predicate.
__m512i select(__m512i words, int shift, bool negate) {
  auto bits = _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                               _mm512_set1_epi32(0xffff));
  return _mm512_xor_si512(bits, _mm512_set1_epi32(negate ? 0x8000 : 0));
}

// Widen low BF16 halfwords into FP32 vector values.
__m512 widen(__m512i bits) { return _mm512_castsi512_ps(_mm512_slli_epi32(bits, 16)); }

} // namespace

// Two FP32 FMAs retain gradual underflow without eligibility checks. On Zen4,
// this is faster than DPBF16 plus guards/repair for its fixed DAZ/FTZ behavior.
void dot2_bf16_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto aw = _mm512_loadu_si512(a + lane), bw = _mm512_loadu_si512(b + lane);
    auto a0 = select(aw, mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0, mode & RJ_ARITH_DOT_NEG_LO_A);
    auto a1 = select(aw, mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16, mode & RJ_ARITH_DOT_NEG_HI_A);
    auto b0 = select(bw, mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0, mode & RJ_ARITH_DOT_NEG_LO_B);
    auto b1 = select(bw, mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16, mode & RJ_ARITH_DOT_NEG_HI_B);
    auto cb = _mm512_xor_si512(_mm512_loadu_si512(c + lane),
                               _mm512_set1_epi32(mode & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0));
    auto acc = _mm512_castsi512_ps(cb);
    auto value = _mm512_fmadd_ps(widen(a0), widen(b0), acc);
    value = _mm512_fmadd_ps(widen(a1), widen(b1), value);
    _mm512_mask_storeu_ps(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

} // namespace rocjitsu::arith
