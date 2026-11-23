// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse gfx12 DOT4 over eight output columns using the dense empirical reduction.
#include "wmma/swmmac_exact16.h"
#include "wmma/x86_64/swmmac_exact16_x86_64.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

namespace rocjitsu::arith {
namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half {

template <bool Bf16, bool Packed> struct HalfStage {
  using V = __m256i;
  using Mask = V;
  using Factor = wmma_v3::Factor;
  static constexpr int Inf = Bf16 ? 0x7f80 : 0x7c00;
  static constexpr int Bias = wmma_v3::kGridBias<Bf16, Packed>;

  static Factor factor(V bits, int bias) { return wmma_v3::decode<Bf16, false>(bits, bias); }
  static V initial(V word) { return word; }

  static bool any(Mask mask) { return exact_dot::lane_mask(mask); }
  static V finite(const Factor (&left)[4], const Factor (&right)[4], V acc, bool saturate,
                  Mask &special) {
    return wmma_v3::stage<Bf16, false, Packed>(left, right, acc, saturate, special);
  }

  static V repair(const Factor (&left)[4], const Factor (&right)[4], V acc, V result,
                  Mask special) {
    using namespace exact_dot;
    auto factor_max = selected_max<V>(left, right);
    special = bit_or(special, greater(factor_max, wmma_v3::splat(Inf - 1)));
    return choose(special, wmma_v3::special<Bf16, false, Packed>(left, right, factor_max, acc),
                  result);
  }
};

} // namespace RJ_ARITH_IMPL_NAMESPACE::sparse_half

template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v3(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index) {
  sparse_half::tile<sparse_half::HalfStage<Bf16, Packed>, WaveSize, Packed>(mode, saturate, dest, a,
                                                                            b, index);
}

template void swmmac_exact16_x86_64_v3<false, 32, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 32, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 64, false>(uint32_t, bool, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<false, 64, true>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 32, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 32, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 64, false>(uint32_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
template void swmmac_exact16_x86_64_v3<true, 64, true>(uint32_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);

} // namespace rocjitsu::arith
