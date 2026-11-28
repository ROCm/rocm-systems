// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 Loose FP32 sine/cosine with vector source/output policy and masked stores.
#include "math/x86_64/trig_loose_x86_64_v4.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "math/trig.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {
// Apply validated trig controls without whole-wave staging or exceptional-lane fallback.
template <bool Cosine>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const auto keep = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  const auto flip = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 16) {
    auto bits = _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(a + lane), keep), flip);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      bits = _mm512_castps_si512(flush_denorm_f32(_mm512_castsi512_ps(bits)));
    }
    auto value = trig_value<Cosine>(bits);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      value = flush_denorm_f32(value);
    }
    if (mode & RJ_ARITH_ALU_OMOD_MASK) {
      value = prepare_omod_f32(value, mode);
      value = _mm512_mul_ps(value, _mm512_set1_ps(scales[(mode >> 6) & 3]));
      const auto magnitude =
          _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MAX));
      const auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800000));
      auto zero = Cosine ? _mm512_setzero_ps()
                         : _mm512_and_ps(value, _mm512_castsi512_ps(_mm512_set1_epi32(INT32_MIN)));
      value = _mm512_mask_mov_ps(value, tiny, zero);
    }
    if (!(flags & RJ_ARITH_FP_IEEE_MODE)) {
      const auto nan = _mm512_cmpgt_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX)),
                                               _mm512_set1_epi32(0x7f800000));
      value = _mm512_mask_mov_ps(value, nan, _mm512_castsi512_ps(bits));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(value));
  }
}
} // namespace

// Evaluate sine or cosine with x86-64-v4, preserving aliases and inactive lanes.
void trig_loose_x86_64_v4(bool cosine, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                          uint32_t *d, const uint32_t *a) {
  if (cosine) {
    return run<true>(flags, exec_mask, mode, d, a);
  }
  return run<false>(flags, exec_mask, mode, d, a);
}
} // namespace rocjitsu::arith
