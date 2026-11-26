// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP32 FRACT with instruction-local guest rounding and staged masked stores.
#include "common/sdwa_output.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "conversion/x86_64/lanes.h"
#include "math/fract.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Compute FP32 fractions, retaining the inherited zero, cap and special-value rules.
__m512 fraction(__m512 x, unsigned rounding, bool quiet_snan) {
  const auto integral = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  __m512 value;
  switch (rounding) {
  case 1:
    value = _mm512_sub_round_ps(x, integral, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    break;
  case 2:
    value = _mm512_sub_round_ps(x, integral, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    break;
  case 3:
    value = _mm512_sub_round_ps(x, integral, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    break;
  default:
    value = _mm512_sub_round_ps(x, integral, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    break;
  }
  value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ),
                             _mm512_setzero_ps());
  const auto limit = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7fffff));
  value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(value, limit, _CMP_GT_OQ), limit);
  const auto bits = _mm512_castps_si512(x);
  const auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  const auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
  const auto special = _mm512_mask_blend_epi32(
      nan, _mm512_set1_epi32(int(0xffc00000U)),
      _mm512_or_si512(bits, _mm512_set1_epi32(quiet_snan ? 0x00400000 : 0)));
  return _mm512_castsi512_ps(_mm512_mask_mov_epi32(
      _mm512_castps_si512(value), _mm512_cmpge_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000)),
      special));
}

// Stage all fractional parts and output controls before publishing overlapping destinations.
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
         const uint32_t *const *a, bool quiet_snan) {
  uint32_t result[1][32];
  const auto rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  for (int lane = 0; lane < 32; lane += 16) {

    auto bits = _mm512_loadu_si512(a[0] + lane);
    bits = _mm512_and_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
    bits = _mm512_xor_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
    auto input = _mm512_castsi512_ps(bits);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      input = flush_denorm_f32(input);
    }
    auto value = fraction(input, rounding, quiet_snan);
    if (mode & RJ_ARITH_ALU_SDWA) {
      // SDWA scales the already-rounded and guest-flushed core result.
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = flush_denorm_f32(value);
      }
      value = _mm512_castsi512_ps(
          sdwa_output_f32(conversion_simd::Lanes<32>(_mm512_castps_si512(value)),
                          sdwa_output_policy(flags, mode))
              .bits);
    } else {
      const unsigned omod = (mode >> 6) & 3;
      if (omod) {
        // OMOD flushes tiny fractions before scaling, so surviving products are exact normals.
        const float scales[] = {1, 2, 4, 0.5f};
        value = _mm512_mul_ps(prepare_omod_f32(value, mode), _mm512_set1_ps(scales[omod]));
      } else if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = flush_denorm_f32(value);
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
      }
    }
    if (!quiet_snan && (!(mode & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
      const auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      value = _mm512_mask_mov_ps(value,
                                 _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000)),
                                 _mm512_castsi512_ps(bits));
    }
    _mm512_storeu_si512(result[0] + lane, _mm512_castps_si512(value));
  }
  for (int word = 0; word < 1; ++word) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[word] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[word] + lane));
    }
  }
}

} // namespace

// Select the validated source width without changing the host FP environment.
void fract_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, bool quiet_snan) {
  run(flags, exec_mask, mode, d, a, quiet_snan);
}

} // namespace rocjitsu::arith
