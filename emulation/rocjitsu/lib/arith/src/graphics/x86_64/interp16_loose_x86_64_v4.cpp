// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native AVX-512 mixed interpolation; exact arithmetic retains its existing model.
#include "common/compiler.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "graphics/interp.h"
#include "rj_arith/detail/rj_arith_graphics.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a Loose quad stage, preserving inactive lanes and the other output half.
// Controls are validated; operands may share whole-register storage.
template <bool P2, bool Rtz>
void interp16_loose_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  RJ_ARITH_UNROLL(2)
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = half_input(_mm512_loadu_si512(a + lane), mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
    x = _mm512_permute_ps(x, P2 ? _MM_SHUFFLE(2, 2, 2, 2) : _MM_SHUFFLE(1, 1, 1, 1));
    auto y = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_loadu_si512(b + lane),
                         _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0)));
    auto words = _mm512_loadu_si512(c + lane);
    __m512 z;
    if constexpr (P2) {
      z = _mm512_castsi512_ps(
          _mm512_xor_si512(words, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0)));
    } else {
      z = _mm512_permute_ps(half_input(words, mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0, mode >> 2),
                            _MM_SHUFFLE(0, 0, 0, 0));
    }
    // RTZ is instruction-local; no host FP controls are changed.
    auto value = Rtz ? _mm512_fmadd_round_ps(x, y, z, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
                     : _mm512_fmadd_ps(x, y, z);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto original = value;
      value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(original, original, _CMP_UNORD_Q),
                                   original);
      }
    }
    auto result = _mm512_castps_si512(value);
    if constexpr (P2) {
      result = half_narrow(value, flags & RJ_ARITH_FP16_OVFL, Rtz ? 3 : 0);
      auto shift = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
      result =
          _mm512_or_si512(_mm512_sll_epi32(result, shift),
                          _mm512_andnot_si512(_mm512_sll_epi32(_mm512_set1_epi32(0xffff), shift),
                                              _mm512_loadu_si512(d + lane)));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void interp16_loose_x86_64_v4<false, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *,
                                                     const uint32_t *);
template void interp16_loose_x86_64_v4<false, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *,
                                                    const uint32_t *);
template void interp16_loose_x86_64_v4<true, false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *,
                                                    const uint32_t *);
template void interp16_loose_x86_64_v4<true, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                   const uint32_t *, const uint32_t *,
                                                   const uint32_t *);
} // namespace rocjitsu::arith
