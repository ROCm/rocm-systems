// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx12 FP8/BF8 WMMA in sixteen-column vectors.
#include "common/compiler.h"
#include "common/simd.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_exact8_x86_64.h"
#include <immintrin.h>

namespace rocjitsu::arith {
namespace {
// Each dword holds four consecutive K values. Transpose these bytes before
// widening so both half-vector conversions produce contiguous rows or columns.
template <bool Bf8> void decode(float (&tile)[16][16], const uint32_t *const *operand) {
  for (int k = 0; k < 16; k += 4) {
    int reg = wmma_gfx12_packed_local(k) / 4, lane = 16 * (k / 8);

    auto words = _mm512_loadu_si512(operand[reg] + lane);
    const auto shuffle = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    auto groups = _mm512_shuffle_epi8(words, _mm512_broadcast_i32x4(shuffle));
    auto bytes = _mm512_permutexvar_epi32(
        _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), groups);
    RJ_ARITH_UNROLL(2)
    for (int pair = 0; pair < 2; ++pair) {
      __m256i packed = pair ? _mm512_extracti64x4_epi64(bytes, 1) : _mm512_castsi512_si256(bytes);
      // E5M2 shares the binary16 exponent bias. E4M3FN uses three
      // fraction bits and a bias eight lower. Scaling by 2^8 is exact,
      // including subnormals. Convert NaNs to signed quiet NaNs first.
      auto raw = _mm512_cvtepu8_epi16(packed);
      auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi16(127));
      auto sign = _mm512_and_si512(_mm512_slli_epi16(raw, 8), _mm512_set1_epi16(INT16_MIN));
      auto nan = _mm512_cmpgt_epi16_mask(magnitude, _mm512_set1_epi16(Bf8 ? 124 : 126));
      auto bits = _mm512_or_si512(
          sign, _mm512_mask_set1_epi16(_mm512_slli_epi16(magnitude, Bf8 ? 8 : 7), nan, 0x7e00));
      auto low = _mm512_cvtph_ps(_mm512_castsi512_si256(bits));
      auto high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1));
      if constexpr (!Bf8) {
        low = _mm512_mul_ps(low, _mm512_set1_ps(256.f));
        high = _mm512_mul_ps(high, _mm512_set1_ps(256.f));
      }
      _mm512_store_ps(tile[k + 2 * pair], low);
      _mm512_store_ps(tile[k + 2 * pair + 1], high);
    }
  }
}

} // namespace
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v4(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  alignas(64) float left[16][16];
  alignas(64) float right[16][16];
  decode<Bf8A>(left, a);
  decode<Bf8B>(right, b);
  __m512 acc[16];
  for (int row = 0; row < 16; ++row) {
    acc[row] = _mm512_castsi512_ps(_mm512_loadu_si512(c[row % 8] + 16 * (row / 8)));
  }
  for (int k = 0; k < 16; ++k) {
    RJ_ARITH_UNROLL(16)
    for (int r = 0; r < 16; ++r) {
      acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(left[k][r]), _mm512_load_ps(right[k]), acc[r]);
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    _mm512_storeu_si512(d[reg], _mm512_castps_si512(acc[reg]));
    _mm512_storeu_si512(d[reg] + 16, _mm512_castps_si512(acc[reg + 8]));
  }
}
template void fp8_wmma_x86_64_v4<false, false>(uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v4<false, true>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v4<true, false>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v4<true, true>(uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);

void fp8_wmma_exact_x86_64_v4(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  wmma_exact8::run(bf8_a, bf8_b, d, a, b, c);
}

} // namespace rocjitsu::arith
