// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 per-lane conditional selection between two source registers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/compiler.h"
#include "register/cndmask.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Half>
void cndmask_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, uint32_t condition) {
  const uint32_t sign = Half ? 0x8000 : 0x80000000;
  const unsigned sa = Half && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
  const unsigned sb = Half && (mode & RJ_ARITH_ALU_HIGH_B) ? 16 : 0;
  const unsigned sd = Half && (mode & RJ_ARITH_ALU_HIGH_D) ? 16 : 0;
  auto clear_a = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX));
  auto clear_b = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_B ? ~sign : UINT32_MAX));
  auto flip_a = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0));
  auto flip_b = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_B ? sign : 0));
  RJ_ARITH_UNROLL(4)
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    if constexpr (Half) {
      x = _mm256_srl_epi32(x, _mm_cvtsi32_si128(sa));
      y = _mm256_srl_epi32(y, _mm_cvtsi32_si128(sb));
    }
    x = _mm256_xor_si256(_mm256_and_si256(x, clear_a), flip_a);
    y = _mm256_xor_si256(_mm256_and_si256(y, clear_b), flip_b);
    auto select = _mm256_sllv_epi32(_mm256_set1_epi32(int(condition >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    auto value = _mm256_blendv_epi8(x, y, _mm256_srai_epi32(select, 31));
    if constexpr (Half) {
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      value = _mm256_sll_epi32(_mm256_and_si256(value, _mm256_set1_epi32(65535)),
                               _mm_cvtsi32_si128(sd));
      value =
          _mm256_or_si256(value, _mm256_andnot_si256(_mm256_set1_epi32(int(65535U << sd)), old));
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

template void cndmask_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                      const uint32_t *, uint32_t);
template void cndmask_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, uint32_t);

} // namespace rocjitsu::arith
