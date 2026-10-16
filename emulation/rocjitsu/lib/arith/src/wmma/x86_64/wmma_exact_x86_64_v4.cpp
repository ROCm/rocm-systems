// SPDX-License-Identifier: MIT

// AVX-512 instantiations of the shared bit-exact FP16/BF16 WMMA algorithm.
// Selects sixteen-column vectors for both wave sizes, gfx11/gfx12 layouts, and
// FP32 or packed results; the algorithm lives in wmma_exact_x86_64.h.

#include "wmma/x86_64/wmma_exact.h"
#include "wmma/x86_64/wmma_exact_x86_64.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated mode; D may alias inputs. saturate applies
// to finite FP16 overflow, and packed gfx11 preserves the unselected D half.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  wmma_exact<Bf16, Lanes, Rdna3, Packed, __m512i>(mode, saturate, d, a, b, c);
}

#define RJ_ARITH_INSTANTIATE(B, L, R, P)                                                           \
  template void wmma_exact_x86_64_v4<B, L, R, P>(uint32_t, bool, uint32_t *const *,                \
                                                 const uint32_t *const *, const uint32_t *const *, \
                                                 const uint32_t *const *);
RJ_ARITH_INSTANTIATE(false, 32, false, false)
RJ_ARITH_INSTANTIATE(false, 64, false, false)
RJ_ARITH_INSTANTIATE(true, 32, false, false)
RJ_ARITH_INSTANTIATE(true, 64, false, false)
RJ_ARITH_INSTANTIATE(false, 32, true, false)
RJ_ARITH_INSTANTIATE(false, 64, true, false)
RJ_ARITH_INSTANTIATE(true, 32, true, false)
RJ_ARITH_INSTANTIATE(true, 64, true, false)
RJ_ARITH_INSTANTIATE(false, 32, false, true)
RJ_ARITH_INSTANTIATE(false, 64, false, true)
RJ_ARITH_INSTANTIATE(true, 32, false, true)
RJ_ARITH_INSTANTIATE(true, 64, false, true)
RJ_ARITH_INSTANTIATE(false, 32, true, true)
RJ_ARITH_INSTANTIATE(false, 64, true, true)
RJ_ARITH_INSTANTIATE(true, 32, true, true)
RJ_ARITH_INSTANTIATE(true, 64, true, true)
#undef RJ_ARITH_INSTANTIATE

} // namespace rocjitsu::arith
