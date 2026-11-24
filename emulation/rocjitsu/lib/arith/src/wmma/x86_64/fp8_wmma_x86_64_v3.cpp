// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 gfx12 FP8/BF8 WMMA kernels using the Wave32 register layout.
// Decodes packed factors once, reuses them across FP32 FMA accumulation, and
// buffers results before stores to support overlapping source/destination VGPRs.

#include "common/compiler.h"
#include "common/simd.h"
#include "wmma/layout.h"
#include "wmma/x86_64/wmma_exact8_x86_64.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Each dword holds four consecutive K values. Transpose these bytes before
// widening so both half-vector conversions produce contiguous rows or columns.
template <bool Bf8> void decode(float (&tile)[16][16], const uint32_t *const *operand) {
  for (int k = 0; k < 16; k += 4) {
    int reg = wmma_gfx12_packed_local(k) / 4, lane = 16 * (k / 8);
    for (int col = 0; col < 16; col += 8) {
      auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(operand[reg] + lane + col));
      const auto shuffle = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
      auto groups = _mm256_shuffle_epi8(words, _mm256_broadcastsi128_si256(shuffle));
      auto bytes = _mm256_permutevar8x32_epi32(groups, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
      RJ_ARITH_UNROLL(2)
      for (int pair = 0; pair < 2; ++pair) {
        __m128i packed = pair ? _mm256_extracti128_si256(bytes, 1) : _mm256_castsi256_si128(bytes);
        // E5M2 shares the binary16 exponent bias. E4M3FN uses three
        // fraction bits and a bias eight lower. Scaling by 2^8 is exact,
        // including subnormals. Convert NaNs to signed quiet NaNs first.
        auto raw = _mm256_cvtepu8_epi16(packed);
        auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi16(127));
        auto sign = _mm256_and_si256(_mm256_slli_epi16(raw, 8), _mm256_set1_epi16(INT16_MIN));
        auto nan = _mm256_cmpgt_epi16(magnitude, _mm256_set1_epi16(Bf8 ? 124 : 126));
        auto bits =
            _mm256_or_si256(sign, _mm256_blendv_epi8(_mm256_slli_epi16(magnitude, Bf8 ? 8 : 7),
                                                     _mm256_set1_epi16(0x7e00), nan));
        auto low = _mm256_cvtph_ps(_mm256_castsi256_si128(bits));
        auto high = _mm256_cvtph_ps(_mm256_extracti128_si256(bits, 1));
        if constexpr (!Bf8) {
          low = _mm256_mul_ps(low, _mm256_set1_ps(256.f));
          high = _mm256_mul_ps(high, _mm256_set1_ps(256.f));
        }
        _mm256_store_ps(tile[k + 2 * pair] + col, low);
        _mm256_store_ps(tile[k + 2 * pair + 1] + col, high);
      }
    }
  }
}

// Compute loose FP8/BF8 A*B+C into FP32 D in gfx12 Wave32 layout.
// D may overlap inputs; all source values are consumed before stores.
template <bool Bf8A, bool Bf8B>
void wmma(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
          const uint32_t *const *c) {
  // Decode each packed factor once, then reuse across the full output matrix.
  alignas(32) float left[16][16], right[16][16];
  decode<Bf8A>(left, a);
  decode<Bf8B>(right, b);
  uint32_t result[8][32];
  // Four rows give eight independent FMA chains across the two column vectors.
  constexpr int Rows = 4;
  for (int base = 0; base < 16; base += Rows) {
    __m256 acc[Rows][2];
    for (int r = 0; r < Rows; ++r) {
      int row = base + r;
      for (int h = 0; h < 2; ++h) {
        acc[r][h] = _mm256_castsi256_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(c[row % 8] + 16 * (row / 8) + 8 * h)));
      }
    }
    RJ_ARITH_UNROLL(4)
    for (int k = 0; k < 16; ++k) {
      auto bl = _mm256_load_ps(right[k]), bh = _mm256_load_ps(right[k] + 8);
      RJ_ARITH_UNROLL(4)
      for (int r = 0; r < Rows; ++r) {
        auto x = _mm256_set1_ps(left[k][base + r]);
        acc[r][0] = _mm256_fmadd_ps(x, bl, acc[r][0]);
        acc[r][1] = _mm256_fmadd_ps(x, bh, acc[r][1]);
      }
    }
    for (int r = 0; r < Rows; ++r) {
      int row = base + r;
      for (int h = 0; h < 2; ++h) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[row % 8] + 16 * (row / 8) + 8 * h),
                            _mm256_castps_si256(acc[r][h]));
      }
    }
  }
  // Snapshot all inputs and results before writes, including cross-register aliases.
  for (int lane = 0; lane < 32; lane += 8) {
    for (int reg = 0; reg < 8; ++reg) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(d[reg] + lane),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
  }
}

} // namespace

// Compute loose gfx12 Wave32 A*B+C into FP32 D for the selected factor formats.
// D may overlap source registers.
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  wmma<Bf8A, Bf8B>(d, a, b, c);
}

template void fp8_wmma_x86_64_v3<false, false>(uint32_t *const *, const uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<false, true>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<true, false>(uint32_t *const *, const uint32_t *const *,
                                              const uint32_t *const *, const uint32_t *const *);
template void fp8_wmma_x86_64_v3<true, true>(uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *);

void fp8_wmma_exact_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  wmma_exact8::run(bf8_a, bf8_b, d, a, b, c);
}

} // namespace rocjitsu::arith
