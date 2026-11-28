// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 Loose FP32/FP16 TANH, including modifiers, half selectors and masked publication.
#include "common/x86_64/alu_x86_64_v3.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "math/x86_64/tanh_loose.h"
#include "math/x86_64/tanh_loose_impl.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compute TANH without staging whole waves; preserve aliases and inactive storage.
template <bool Half>
void tanh_loose_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const int source_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned omod = (mode >> 6) & 3;
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 8) {
    auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i raw;
    if constexpr (Half) {
      raw = _mm256_castps_si256(half_input<false>(words, source_shift, mode));
    } else {
      raw = _mm256_xor_si256(
          _mm256_and_si256(words, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1)),
          _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
    }
    auto value = _mm256_castsi256_ps((__m256i)tanh_vector<F8, I8>((I8)raw));
    if constexpr (Half) {
      // OMOD consumes a rounded half result, as in the inherited output policy.
      if (omod) {
        value = half_input<false>(half_narrow<false>(value, false), 0, 0);
        value = prepare_omod_f16(value, mode);
      }
    } else if (omod) {
      value = prepare_omod_f32(value, mode);
    }
    if (omod) {
      value = _mm256_mul_ps(value, _mm256_set1_ps(scales[omod]));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, false);
    }
    __m256i result;
    if constexpr (Half) {
      result =
          _mm256_sll_epi32(half_narrow<false>(value, false), _mm_cvtsi32_si128(destination_shift));
      result = _mm256_or_si256(
          result,
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane)),
                           _mm256_set1_epi32(destination_shift ? 0xffff : int(0xffff0000U))));
    } else {
      result = _mm256_castps_si256(value);
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void tanh_loose_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void tanh_loose_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
