// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 DOT4 over 16 output columns using the dense empirical reduction.
#include "wmma/swmmac_exact16.h"
#include "wmma/x86_64/swmmac_exact16_x86_64.h"
#include "wmma/x86_64/wmma_dot_x86_64_v4.h"

namespace rocjitsu::arith {
namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half {

template <bool Bf16, bool Packed> struct HalfStage {
  using V = __m512i;
  using Mask = __mmask16;
  using Factor = wmma_v4::WmmaFactor;
  static constexpr int Inf = Bf16 ? 0x7f80 : 0x7c00;
  static constexpr int Bias = wmma_v4::kWmmaGridBias<Bf16, Packed>;

  static Factor factor(V bits, int bias) { return wmma_v4::wmma_factor<Bf16, false>(bits, bias); }
  static V initial(V word) { return wmma_v4::wmma_extend<Packed>(word); }

  static bool any(Mask mask) { return mask != 0; }
  static V finite(const Factor (&left)[4], const Factor (&right)[4], V acc, bool saturate,
                  Mask &special) {
    return wmma_v4::wmma_dot<Bf16, false, Packed>(left, right, acc, saturate, special);
  }

  static V repair(const Factor (&left)[4], const Factor (&right)[4], V acc, V result,
                  Mask special) {
    auto factor_max = selected_max<V>(left, right);
    special |= _mm512_cmpge_epi32_mask(factor_max, wmma_v4::splat(Inf));
    return _mm512_mask_mov_epi32(
        result, special, wmma_v4::wmma_special<Bf16, false, Packed>(left, right, factor_max, acc));
  }
};

} // namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half

template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index) {
  sparse_half::tile<sparse_half::HalfStage<Bf16, Packed>, WaveSize, Packed>(mode, saturate, dest, a,
                                                                            b, index);
}

template void swmmac_exact16_x86_64_v4<false, 32, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 32, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 64, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<false, 64, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 32, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 32, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 64, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v4<true, 64, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
