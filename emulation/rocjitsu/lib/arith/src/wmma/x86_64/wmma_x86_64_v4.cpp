// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose gfx11/gfx12 half WMMA. Each vector holds one full output row.
#include "common/compiler.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include "wmma/x86_64/wmma_loose_x86_64_v4.h"

#include <immintrin.h>

namespace rocjitsu::arith {
namespace {

template <bool Bf16, int Lanes, bool Rdna3, bool Packed, bool Saturate>
void compute(uint32_t mods, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
             const uint32_t *const *c) {
  const auto a_sign = _mm512_set1_epi32((mods & RJ_ARITH_WMMA_NEG_LO_A ? 0x8000 : 0) |
                                        (mods & RJ_ARITH_WMMA_NEG_HI_A ? INT32_MIN : 0));
  const auto b_sign = _mm512_set1_epi32((mods & RJ_ARITH_WMMA_NEG_LO_B ? 0x8000 : 0) |
                                        (mods & RJ_ARITH_WMMA_NEG_HI_B ? INT32_MIN : 0));

  // Pre-decode A into K-major memory so vfmadd231ps folds {1to16} broadcasts
  // directly from stack without STLF stalls (separated by the C load pass).
  // For FP32 and BF16 outputs, decode B on-the-fly per K-pair to avoid a 1 KB
  // right[16][16] buffer while keeping acc[16] spill-free.
  alignas(64) float left[16][16];
  for (int step = 0; step < 16; step += 2) {
    const int k = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(step) : step;
    const WmmaRegisterSection src = wmma_half_input<Lanes, Rdna3>(k);
    const __m512i aw = _mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane), a_sign);
    _mm512_store_ps(left[step], wmma_widen_half<Bf16>(aw, 0));
    _mm512_store_ps(left[step + 1], wmma_widen_half<Bf16>(aw, 16));
  }

  __m512 acc[16];
  wmma_load_acc<Bf16, Lanes, Rdna3, Packed>(acc, mods, c);

  auto fma_pass = [&](const float *a_row, __m512 b_vec) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      acc[row] = _mm512_fmadd_ps(_mm512_set1_ps(a_row[row]), b_vec, acc[row]);
    }
  };

  auto load_b_pair = [&](int pair, __m512 &b0, __m512 &b1) {
    const uint32_t *ptr;
    if constexpr (Rdna3) {
      ptr = b[pair];
    } else if constexpr (Lanes == 64) {
      ptr = b[pair & 1] + 16 * (pair >> 1);
    } else if constexpr (Packed) {
      ptr = b[(pair & 1) + 2 * (pair >> 2)] + 16 * ((pair >> 1) & 1);
    } else {
      ptr = b[pair & 3] + 16 * (pair >> 2);
    }
    const __m512i bw = _mm512_xor_si512(_mm512_loadu_si512(ptr), b_sign);
    b0 = wmma_widen_half<Bf16>(bw, 0);
    b1 = wmma_widen_half<Bf16>(bw, 16);
  };

  // Keep A in K-major memory so broadcasts fold into FMA without extra shuffles.
  RJ_ARITH_UNROLL(1)
  for (int pair = 0; pair < 8; ++pair) {
    __m512 b0, b1;
    load_b_pair(pair, b0, b1);
    fma_pass(left[2 * pair], b0);
    fma_pass(left[2 * pair + 1], b1);
  }

  wmma_store_acc<Bf16, Lanes, Rdna3, Packed, Saturate>(acc, mods, d);
}

} // namespace

template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_loose_x86_64_v4(uint32_t mods, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  if constexpr (Packed && !Bf16) {
    if (saturate) {
      compute<Bf16, Lanes, Rdna3, Packed, true>(mods, d, a, b, c);
      return;
    }
  }
  compute<Bf16, Lanes, Rdna3, Packed, false>(mods, d, a, b, c);
}

template void wmma_loose_x86_64_v4<false, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *,
                                                            const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<false, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 32, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 32, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 32, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 32, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 64, false, false>(uint32_t, bool, uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *,
                                                           const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 64, false, true>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 64, true, false>(uint32_t, bool, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *);
template void wmma_loose_x86_64_v4<true, 64, true, true>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);

} // namespace rocjitsu::arith
