// SPDX-License-Identifier: MIT

// AVX2 loose gfx12 FP8/BF8 WMMA kernel using the Wave32 register layout.
// Decodes packed factors once, reuses them across FP32 FMA accumulation, and
// buffers results before stores to support overlapping source/destination VGPRs.

#include "common/simd.h"
#include "common/x86_64/fp8.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen a gfx12 Wave32 FP8/BF8 factor tile into values[k][row_or_column].
template <bool Bf8> void decode(float (&values)[16][16], const uint32_t *const *v) {
  for (int k = 0; k < 16; ++k) {
    for (int index = 0; index < 16; index += 8) {
      auto words = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(v[(k % 8) / 4] + index + 16 * (k / 8)));
      words = _mm256_srl_epi32(words, _mm_cvtsi32_si128(8 * (k % 4)));
      _mm256_storeu_ps(values[k] + index, widen_fp8<Bf8>(words));
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
  for (int row = 0; row < 16; ++row) {
    const int reg = row % 8, lane = 16 * (row / 8);
    // Captured C ABS/NEG encodings are ignored by FP8/BF8 WMMA.
    auto low =
        _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane)));
    auto high = _mm256_castsi256_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane + 8)));
    for (int k = 0; k < 16; ++k) {
      auto x = _mm256_set1_ps(left[k][row]);
      low = _mm256_fmadd_ps(x, _mm256_load_ps(right[k]), low);
      high = _mm256_fmadd_ps(x, _mm256_load_ps(right[k] + 8), high);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane), _mm256_castps_si256(low));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane + 8),
                        _mm256_castps_si256(high));
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

// Compute loose gfx12 Wave32 A*B+C into FP32 D. bf8_a/b select each
// factor's format independently; D may overlap source registers.
void fp8_wmma_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  if (bf8_a) {
    if (bf8_b) {
      wmma<true, true>(d, a, b, c);
    } else {
      wmma<true, false>(d, a, b, c);
    }
  } else {
    if (bf8_b) {
      wmma<false, true>(d, a, b, c);
    } else {
      wmma<false, false>(d, a, b, c);
    }
  }
}

} // namespace rocjitsu::arith
