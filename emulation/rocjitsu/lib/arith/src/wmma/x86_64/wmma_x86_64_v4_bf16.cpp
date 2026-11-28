// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 pair-dot backend for loose gfx11/gfx12 BF16 WMMA, including
// both wave sizes and FP32 or packed BF16 results. Native pair-dot rounding
// and flushing are accepted throughout the Loose input domain.

#include "common/compiler.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compute Loose BF16 WMMA with validated modifiers, permitting D to alias inputs.
template <int Lanes, bool Rdna3, bool Packed>
void wmma_x86_64_v4_bf16(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c) {
  const __m512i a_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const __m512i b_sign =
      _mm512_set1_epi32((instruction_flags & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                        (instruction_flags & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));

  alignas(64) uint32_t left[8][16];
  alignas(64) uint32_t right[8][16];
  for (int step = 0; step < 8; ++step) {
    const int pair = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(2 * step) / 2 : step;
    const rocjitsu::arith::WmmaRegisterSection src = wmma_half_input<Lanes, Rdna3>(2 * pair);
    const __m512i aw_rot =
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane), a_sign), 16);
    const __m512i bw_rot =
        _mm512_rol_epi32(_mm512_xor_si512(_mm512_loadu_si512(b[src.reg] + src.lane), b_sign), 16);
    _mm512_store_si512(left[step], aw_rot);
    _mm512_store_si512(right[step], bw_rot);
  }
  // Native BF16 flushing and pair-dot rounding are permitted in Loose.
  __m512 sums[16];
  wmma_load_acc<true, Lanes, Rdna3, Packed>(sums, instruction_flags, c);
  wmma_bf16_pairs(sums, left, right);
  wmma_store_acc<true, Lanes, Rdna3, Packed, false>(sums, instruction_flags, d);
}

template void wmma_x86_64_v4_bf16<32, false, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_x86_64_v4_bf16<32, false, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_x86_64_v4_bf16<32, true, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_x86_64_v4_bf16<32, true, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);
template void wmma_x86_64_v4_bf16<64, false, false>(uint32_t, uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template void wmma_x86_64_v4_bf16<64, false, true>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_x86_64_v4_bf16<64, true, false>(uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void wmma_x86_64_v4_bf16<64, true, true>(uint32_t, uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *);

} // namespace rocjitsu::arith
