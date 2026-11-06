// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 interpolation with packed source selection and staged rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/x86_64/mixed_half_x86_64_v3.h"
#include "graphics/interp.h"
#include "graphics/interp16_scalar.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate an FP16 interpolation stage. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool P2, bool Rtz>
void interp16_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto aw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto x = half_input<false>(aw, mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
    x = _mm256_permute_ps(x, P2 ? _MM_SHUFFLE(2, 2, 2, 2) : _MM_SHUFFLE(1, 1, 1, 1));
    auto y = _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                         _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0)));
    auto cw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    __m256 z;
    if constexpr (P2) {
      z = _mm256_castsi256_ps(
          _mm256_xor_si256(cw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0)));
    } else {
      z = _mm256_permute_ps(half_input<false>(cw, mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0, mode >> 2),
                            _MM_SHUFFLE(0, 0, 0, 0));
    }
    __m256i result;
    if constexpr (P2) {
      auto lo =
          mixed_half_value4<Rtz>(_mm256_castps256_ps128(x), _mm256_castps256_ps128(y),
                                 _mm256_castps256_ps128(z), saturate, mode & RJ_ARITH_ALU_CLAMP);
      auto hi =
          mixed_half_value4<Rtz>(_mm256_extractf128_ps(x, 1), _mm256_extractf128_ps(y, 1),
                                 _mm256_extractf128_ps(z, 1), saturate, mode & RJ_ARITH_ALU_CLAMP);
      result = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
      unsigned sd = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      result = _mm256_or_si256(_mm256_sll_epi32(result, _mm_cvtsi32_si128(sd)),
                               _mm256_andnot_si256(_mm256_set1_epi32(int(65535u << sd)), old));
    } else {
      __m256 value;
      if constexpr (Rtz) {
        alignas(32) uint32_t xs[8], ys[8], zs[8], results[8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(xs), _mm256_castps_si256(x));
        _mm256_store_si256(reinterpret_cast<__m256i *>(ys), _mm256_castps_si256(y));
        _mm256_store_si256(reinterpret_cast<__m256i *>(zs), _mm256_castps_si256(z));
        for (int i = 0; i < 8; ++i) {
          results[i] = as_bits(interp16_rtz_float(xs[i], ys[i], zs[i]));
        }
        value = _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(results)));
      } else {
        value = _mm256_fmadd_ps(x, y, z);
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      }
      result = _mm256_castps_si256(value);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void interp16_x86_64_v3<false, false>(bool, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *, const uint32_t *,
                                               const uint32_t *);
template void interp16_x86_64_v3<true, false>(bool, uint32_t, uint32_t, uint32_t *,
                                              const uint32_t *, const uint32_t *, const uint32_t *);
template void interp16_x86_64_v3<false, true>(bool, uint32_t, uint32_t, uint32_t *,
                                              const uint32_t *, const uint32_t *, const uint32_t *);
template void interp16_x86_64_v3<true, true>(bool, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
