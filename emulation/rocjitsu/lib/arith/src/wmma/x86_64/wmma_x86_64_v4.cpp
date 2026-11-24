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
  alignas(64) float left[16][16];
  __m512 right[16];
  // Decode each packed pair once in the order consumed by this layout.
  for (int step = 0; step < 16; step += 2) {
    int k = Packed && !Rdna3 && Lanes == 32 ? wmma_gfx12_w32_k(step) : step;
    auto src = wmma_half_input<Lanes, Rdna3>(k);
    auto aw = _mm512_xor_si512(_mm512_loadu_si512(a[src.reg] + src.lane), a_sign);
    auto bw = _mm512_xor_si512(_mm512_loadu_si512(b[src.reg] + src.lane), b_sign);
    _mm512_store_ps(left[step], wmma_widen_half<Bf16>(aw, 0));
    _mm512_store_ps(left[step + 1], wmma_widen_half<Bf16>(aw, 16));
    right[step] = wmma_widen_half<Bf16>(bw, 0);
    right[step + 1] = wmma_widen_half<Bf16>(bw, 16);
  }
  __m512 acc[16];
  wmma_load_acc<Bf16, Lanes, Rdna3, Packed>(acc, mods, c);
  for (int step = 0; step < 16; step += 2) {
    RJ_ARITH_UNROLL(16)
    for (int row = 0; row < 16; ++row) {
      auto fma = [&](int k, __m512 sum) {
        auto factor = _mm512_set1_ps(left[k][row]);
        if constexpr (Rdna3 && Packed) {
          // gfx11 packed arithmetic preserves the host exception flags.
          return _mm512_fmadd_round_ps(factor, right[k], sum,
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        } else {
          return _mm512_fmadd_ps(factor, right[k], sum);
        }
      };
      acc[row] = fma(step + 1, fma(step, acc[row]));
      if constexpr (Packed) {
        if (step < 14 && (Rdna3 || step % 4 == 2)) {
          if constexpr (Bf16) {
            acc[row] = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(acc[row]), _mm512_set1_epi32(0xffff0000u)));
          } else {
            acc[row] =
                _mm512_cvt_roundph_ps(wmma_narrow_f16<Saturate>(acc[row]), _MM_FROUND_NO_EXC);
          }
        }
      }
    }
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
