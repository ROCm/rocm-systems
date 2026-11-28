// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 Loose FP32/FP16 TANH, including modifiers, half selectors and masked publication.
#include "common/x86_64/alu_x86_64_v4.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "math/x86_64/tanh_loose.h"
#include "math/x86_64/tanh_loose_impl.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Compute TANH without staging whole waves; preserve aliases and inactive storage.
template <bool Half>
void tanh_loose_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const int source_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned omod = (mode >> 6) & 3;
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 16) {
    auto words = _mm512_loadu_si512(a + lane);
    __m512i raw;
    if constexpr (Half) {
      raw = _mm512_castps_si512(half_input(words, source_shift, mode));
    } else {
      raw = _mm512_xor_si512(
          _mm512_and_si512(words, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1)),
          _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
    }
    auto value = _mm512_castsi512_ps((__m512i)tanh_vector<F16, I16>((I16)raw));
    if constexpr (Half) {
      // OMOD consumes a rounded half result, as in the inherited output policy.
      if (omod) {
        value = half_input(half_narrow(value, false), 0, 0);
        auto bits = _mm512_castps_si512(value);
        const auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
        const auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x38800000));
        if (omod == 3) {
          const auto underflow = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x39000000));
          bits = _mm512_mask_and_epi32(bits, underflow, bits, _mm512_set1_epi32(INT32_MIN));
        }
        value = _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits, tiny, _mm512_setzero_si512()));
      }
    } else if (omod) {
      value = prepare_omod_f32(value, mode);
    }
    if (omod) {
      value = _mm512_mul_ps(value, _mm512_set1_ps(scales[omod]));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, false);
    }
    __m512i result;
    if constexpr (Half) {
      result = _mm512_sll_epi32(half_narrow(value, false), _mm_cvtsi32_si128(destination_shift));
      result = _mm512_or_si512(
          result,
          _mm512_and_si512(_mm512_loadu_si512(d + lane),
                           _mm512_set1_epi32(destination_shift ? 0xffff : int(0xffff0000U))));
    } else {
      result = _mm512_castps_si512(value);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void tanh_loose_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void tanh_loose_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
