// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v4 dense byte matrix entry points.
#include "common/compiler.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "mfma/backends.h"
#include "mfma/integer.h"
#include "mfma/x86_64/batch4.h"
namespace rocjitsu::arith {
namespace {
// Each lane holds four signed bytes. Multiply alternating byte pairs as signed
// halfwords so their two-product sums retain the full 32-bit range.
void batch4_i8(uint64_t controls, bool clamp, uint32_t *const *d, const uint32_t *a,
               const uint32_t *b, const uint32_t *const *c) {
  __m512i result[4][4];
  for (int lane = 0; lane < 64; lane += 16) {
    int first_a = mfma_a_lane(lane, controls);
    auto groups = mfma_batch4_groups(lane, first_a, controls);
    auto left = _mm512_loadu_si512(a + (first_a & ~15));
    auto right = _mm512_loadu_si512(b + mfma_b_lane(lane, controls));
    auto even = [](__m512i bytes) { return _mm512_srai_epi16(_mm512_slli_epi16(bytes, 8), 8); };
    auto b0 = even(right), b1 = _mm512_srai_epi16(right, 8);
    RJ_ARITH_UNROLL(4)
    for (int row = 0; row < 4; ++row) {
      auto x = _mm512_permutexvar_epi32(_mm512_add_epi32(groups, _mm512_set1_epi32(row)), left);
      auto dot = _mm512_add_epi32(_mm512_madd_epi16(even(x), b0),
                                  _mm512_madd_epi16(_mm512_srai_epi16(x, 8), b1));
      auto initial = _mm512_loadu_si512(c[row] + lane);
      auto sum = _mm512_add_epi32(initial, dot);
      result[row][lane / 16] = clamp ? saturate_signed_sum(initial, dot, sum) : sum;
    }
  }
  for (int row = 0; row < 4; ++row) {
    for (int lane = 0; lane < 64; lane += 16) {
      _mm512_storeu_si512(d[row] + lane, result[row][lane / 16]);
    }
  }
}
} // namespace
// Evaluate the full matrix using this translation unit's SIMD reduction.
template <int M, int K, int Batch, bool Wmma, int ABlockLanes>
void matrix_i8_x86_64_v4(uint64_t controls, bool clamp_overflow, bool clamp_per_pass,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  if constexpr (M == 4 && ABlockLanes == 0) {
    static_assert(K == 4 && Batch == 16 && !Wmma);
    return batch4_i8(controls, clamp_overflow, d, a[0], b[0], c);
  }
  RJ_ARITH_IMPL_NAMESPACE::matrix_i8_kernel<M, K, Batch, Wmma, ABlockLanes>(
      controls, clamp_overflow, clamp_per_pass, d, a, b, c);
}
template void matrix_i8_x86_64_v4<32, 4, 2, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 4, 4, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<4, 4, 16, false>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 16, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    bool clamp_per_pass, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, true>(uint64_t controls, bool clamp_overflow,
                                                   bool clamp_per_pass, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Older full-wave shapes and gfx90a adjacent-block broadcasts share the reduction.
template void matrix_i8_x86_64_v4<32, 4, 2, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 4, 4, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<4, 4, 16, false, 4>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 32>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 0>(uint64_t, bool, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 16>(uint64_t, bool, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith
