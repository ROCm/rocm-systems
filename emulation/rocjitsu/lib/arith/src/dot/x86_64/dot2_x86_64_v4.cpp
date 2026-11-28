// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 loose FP16/BF16 DOT2 with FP32 FMA accumulation and operand-half selection.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract one FP16/BF16 halfword per lane and apply its sign modifier.
__m512i select(__m512i words, int shift, bool negate) {
  auto bits = _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                               _mm512_set1_epi32(0xffff));
  return _mm512_xor_si512(bits, _mm512_set1_epi32(negate ? 0x8000 : 0));
}

// Widen selected FP16/BF16 halfwords into FP32 vector values.
template <bool Bf16> __m512 widen(__m512i bits) {
  if constexpr (Bf16) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(bits, 16));
  } else {
    return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(bits));
  }
}

} // namespace

// Accumulate selected packed products through two FP32 FMAs per lane.
template <int Lanes, bool Bf16>
void dot2_x86_64_v4(uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
                    const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto aw = _mm512_loadu_si512(a + lane), bw = _mm512_loadu_si512(b + lane);
    auto a0 = select(aw, instruction_flags & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0,
                     instruction_flags & RJ_ARITH_DOT_NEG_LO_A);
    auto a1 = select(aw, instruction_flags & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16,
                     instruction_flags & RJ_ARITH_DOT_NEG_HI_A);
    auto b0 = select(bw, instruction_flags & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0,
                     instruction_flags & RJ_ARITH_DOT_NEG_LO_B);
    auto b1 = select(bw, instruction_flags & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16,
                     instruction_flags & RJ_ARITH_DOT_NEG_HI_B);
    auto cb = _mm512_xor_si512(
        _mm512_and_si512(
            _mm512_loadu_si512(c + lane),
            _mm512_set1_epi32(instruction_flags & RJ_ARITH_DOT_ABS_C ? 0x7fffffff : -1)),
        _mm512_set1_epi32(instruction_flags & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0));
    auto acc = _mm512_castsi512_ps(cb);
    auto value = _mm512_fmadd_ps(widen<Bf16>(a0), widen<Bf16>(b0), acc);
    value = _mm512_fmadd_ps(widen<Bf16>(a1), widen<Bf16>(b1), value);
    _mm512_mask_storeu_ps(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

template void dot2_x86_64_v4<32, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                        const uint32_t *, const uint32_t *);
template void dot2_x86_64_v4<32, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *);
template void dot2_x86_64_v4<64, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                        const uint32_t *, const uint32_t *);
template void dot2_x86_64_v4<64, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
