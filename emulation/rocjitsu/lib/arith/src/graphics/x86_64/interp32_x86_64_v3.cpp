// SPDX-License-Identifier: MIT

// AVX2 FP32 interpolation stages with source-lane selection.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "graphics/interp.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate an FP32 interpolation stage. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2>
void interp32_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c) {
  auto flip_a = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto flip_b = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  auto flip_c = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), flip_a));
    auto y = _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), flip_b));
    auto z = _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), flip_c));

    // Each shuffle reads all four source lanes, regardless of EXEC.
    x = _mm256_permute_ps(x, P2 ? _MM_SHUFFLE(2, 2, 2, 2) : _MM_SHUFFLE(1, 1, 1, 1));
    if constexpr (!P2) {
      z = _mm256_permute_ps(z, _MM_SHUFFLE(0, 0, 0, 0));
    }
    auto value = _mm256_fmadd_ps(x, y, z);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

template void interp32_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *);
template void interp32_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                        const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
