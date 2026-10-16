// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 packed FP16 FMA/FMAC with independent low/high operand selectors.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/packed_alu.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/packed_fma.h"
#include "fp/x86_64/half_fma_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Evaluate one packed FP16 FMA result half under the supplied per-half selectors and guest
// controls.
__m256i compute_half(__m256i a, __m256i b, __m256i c, uint64_t flags, uint32_t mode) {
  auto x = half_input(a, mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
  auto y = half_input(b, mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0, mode >> 1);
  auto z = half_input(c, mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0, mode >> 2);
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    // Replace subnormal FP32 lanes with signed zero.
    auto flush = [](__m256 input) {
      auto bits = _mm256_castps_si256(input);
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000),
                                     _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)));
      return _mm256_castsi256_ps(
          _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)), tiny));
    };
    x = flush(x);
    y = flush(y);
    z = flush(z);
  }
  return half_fma_value(x, y, z, flags & RJ_ARITH_FP16_OVFL, mode & RJ_ARITH_ALU_CLAMP,
                        flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Evaluate paired FP16 fused multiply-adds. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Accumulate>
void packed_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  uint32_t low_mode, high_mode;
  if constexpr (Accumulate) {
    low_mode = 0;
    high_mode = low_mode | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C;
  } else {
    low_mode = packed_half_mode(mode, false);
    high_mode = packed_half_mode(mode, true);
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto low = compute_half(va, vb, vc, flags, low_mode);
    auto high = compute_half(va, vb, vc, flags, high_mode);
    auto result = _mm256_or_si256(low, _mm256_slli_epi32(high, 16));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void packed_fma_x86_64_v3<false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                          const uint32_t *, const uint32_t *, const uint32_t *);
template void packed_fma_x86_64_v3<true>(uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                         const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
