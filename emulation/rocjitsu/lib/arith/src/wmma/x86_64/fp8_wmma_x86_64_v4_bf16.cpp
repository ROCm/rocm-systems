// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native BF16 pair dots for loose gfx12 FP8/BF8 WMMA.

#include "common/simd.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

namespace rocjitsu::arith {

// All finite FP8 products are integer multiples of 2^-32.
// A nonzero product cannot cancel a normal FP32 accumulator into a subnormal:
// the spacing near that product is at least 2^-56. A subnormal input C,
// however, would be flushed by DPBF16. Let the general FMA path handle it.
template <bool Bf8A, bool Bf8B>
bool fp8_wmma_x86_64_v4_bf16(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  __m512 acc[16];
  __mmask16 subnormal = 0;
  for (int row = 0; row < 16; ++row) {
    auto bits = _mm512_loadu_si512(c[row % 8] + 16 * (row / 8));
    acc[row] = _mm512_castsi512_ps(bits);
    subnormal |= wmma_f32_subnormal(bits);
  }
  if (subnormal) {
    return false;
  }
  alignas(64) uint32_t left[8][16];
  __m512i right[8];
  for (int k = 0; k < 16; k += 4) {
    int reg = wmma_gfx12_packed_local(k) / 4, lane = 16 * (k / 8);
    auto aw = _mm512_loadu_si512(a[reg] + lane), bw = _mm512_loadu_si512(b[reg] + lane);
    for (int h = 0; h < 2; ++h) {
      _mm512_store_si512(left[k / 2 + h], wmma_fp8_bf16_pairs<Bf8A>(aw, h));
      right[k / 2 + h] = wmma_fp8_bf16_pairs<Bf8B>(bw, h);
    }
  }
  wmma_bf16_pairs<false, false>(acc, left, right);
  wmma_store_acc<true, 32, false, false, false>(acc, 0, d);
  return true;
}

template bool fp8_wmma_x86_64_v4_bf16<false, false>(uint32_t *const *, const uint32_t *const *,
                                                    const uint32_t *const *,
                                                    const uint32_t *const *);
template bool fp8_wmma_x86_64_v4_bf16<false, true>(uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *,
                                                   const uint32_t *const *);
template bool fp8_wmma_x86_64_v4_bf16<true, false>(uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *,
                                                   const uint32_t *const *);
template bool fp8_wmma_x86_64_v4_bf16<true, true>(uint32_t *const *, const uint32_t *const *,
                                                  const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
