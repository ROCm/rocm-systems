// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 half SIN/COS sharing the inherited polynomial and MODE policies.
// Exact numerical modes apply the same half-rounding repair as AVX2.
#include "common/validation.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "math/trig.h"
#include "math/trig_coefficients.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Evaluate the existing AVX2 reduced polynomial at AVX-512 width, before half rounding.
template <bool Cosine> __m512 value(__m512i bits) {
  auto zero = _mm512_setzero_si512();
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  auto mantissa = _mm512_or_si512(_mm512_and_si512(magnitude, _mm512_set1_epi32(0x7fffff)),
                                  _mm512_set1_epi32(0x800000));
  // Variable shifts outside [0,31] produce zero, covering large and tiny inputs.
  auto shift = _mm512_sub_epi32(_mm512_srli_epi32(magnitude, 23), _mm512_set1_epi32(119));
  auto phase =
      _mm512_and_si512(_mm512_or_si512(_mm512_sllv_epi32(mantissa, shift),
                                       _mm512_srlv_epi32(mantissa, _mm512_sub_epi32(zero, shift))),
                       _mm512_set1_epi32(INT32_MAX));
  auto quadrant = _mm512_srli_epi32(phase, 29);
  auto quarter = _mm512_and_si512(phase, _mm512_set1_epi32(0x1fffffff));
  auto reflected = _mm512_cmpgt_epi32_mask(quarter, _mm512_set1_epi32(0x0fffffff));
  auto reduced = _mm512_mask_sub_epi32(quarter, reflected, _mm512_set1_epi32(0x20000000), quarter);
  auto table =
      _mm512_and_si512(_mm512_xor_si512(_mm512_xor_si512(quadrant, _mm512_movm_epi32(reflected)),
                                        _mm512_set1_epi32(Cosine)),
                       _mm512_set1_epi32(1));
  auto index = _mm512_srli_epi32(
      _mm512_mask_sub_epi32(reduced, reflected, reduced, _mm512_set1_epi32(1)), 24);
  auto fraction = _mm512_sub_epi32(reduced, _mm512_slli_epi32(index, 24));
  auto column = _mm512_add_epi32(index, _mm512_slli_epi32(table, 4));
  auto t = _mm512_mul_ps(_mm512_cvtepi32_ps(fraction), _mm512_set1_ps(0x1p-24f));
  auto c0 = _mm512_cvtepi32_ps(_mm512_i32gather_epi32(column, trig::coefficients[0], 4));
  auto c1 = _mm512_cvtepi32_ps(_mm512_i32gather_epi32(column, trig::coefficients[1], 4));
  auto c2 = _mm512_cvtepi32_ps(_mm512_i32gather_epi32(column, trig::coefficients[2], 4));
  auto c3 = _mm512_cvtepi32_ps(_mm512_i32gather_epi32(column, trig::coefficients[3], 4));
  auto result =
      _mm512_mul_ps(_mm512_fmadd_ps(t, _mm512_fmadd_ps(t, _mm512_fmadd_ps(t, c3, c2), c1), c0),
                    _mm512_set1_ps(0x1p-26f));
  auto sign = _mm512_srli_epi32(quadrant, 1);
  if constexpr (Cosine) {
    sign = _mm512_xor_si512(sign, quadrant);
  } else {
    sign = _mm512_xor_si512(sign, _mm512_srli_epi32(bits, 31));
  }
  sign = _mm512_mask_mov_epi32(_mm512_slli_epi32(sign, 31),
                               _mm512_cmp_ps_mask(result, _mm512_setzero_ps(), _CMP_EQ_OQ), zero);
  result = _mm512_xor_ps(result, _mm512_castsi512_ps(sign));
  if constexpr (!Cosine) {
    auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x39c00000));
    auto small = _mm512_castsi512_ps(_mm512_maskz_mov_epi32(tiny, bits));
    small = _mm512_mul_ps(small, _mm512_castsi512_ps(_mm512_set1_epi32(0x40c90fd5)));
    result = _mm512_mask_mov_ps(result, tiny, small);
  }
  auto infinity = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
  auto nan = _mm512_mask_mov_epi32(_mm512_or_si512(bits, _mm512_set1_epi32(0x400000)), infinity,
                                   _mm512_set1_epi32(0xffc00000U));
  return _mm512_mask_mov_ps(result,
                            _mm512_cmpge_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000)),
                            _mm512_castsi512_ps(nan));
}

// Apply half input/output policy and publish only the selected active halves.
template <bool Cosine>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned omod = (mode >> 6) & 3;
  const __mmask16 exact = rocjitsu::arith::is_exact(flags) ? 0xffff : 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto words = _mm512_loadu_si512(a + lane);
    auto raw = _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(a_shift)),
                                _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    raw = _mm512_xor_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
    auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7fff));
    auto x = half_input(words, a_shift, mode);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
      x = _mm512_castsi512_ps(_mm512_mask_and_epi32(
          _mm512_castps_si512(x), tiny, _mm512_castps_si512(x), _mm512_set1_epi32(INT32_MIN)));
    }
    auto result = half_narrow(value<Cosine>(_mm512_castps_si512(x)), false);
    // Only these two signed inputs per operation need a one-half-ULP repair
    // to equal the inherited integer model; Loose shares the polynomial.
    auto repair =
        _mm512_mask_cmpeq_epi32_mask(exact, magnitude, _mm512_set1_epi32(Cosine ? 0x3251 : 0x2abc));
    result = _mm512_mask_sub_epi32(result, repair, result, _mm512_set1_epi32(1));
    if (!(flags & RJ_ARITH_FP_IEEE_MODE)) {
      result = _mm512_mask_mov_epi32(
          result, _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7c00)), raw);
    }
    magnitude = _mm512_and_si512(result, _mm512_set1_epi32(0x7fff));
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      result = _mm512_mask_and_epi32(result,
                                     _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400)),
                                     result, _mm512_set1_epi32(0x8000));
      magnitude = _mm512_and_si512(result, _mm512_set1_epi32(0x7fff));
    }
    if (omod) {
      auto scaled = _mm512_add_epi32(magnitude, _mm512_set1_epi32(omod << 10));
      if (omod == 3) {
        auto rounded = _mm512_add_epi32(
            _mm512_srli_epi32(magnitude, 1),
            _mm512_maskz_set1_epi32(
                _mm512_cmpeq_epi32_mask(_mm512_and_si512(magnitude, _mm512_set1_epi32(3)),
                                        _mm512_set1_epi32(3)),
                1));
        scaled = _mm512_mask_mov_epi32(_mm512_sub_epi32(magnitude, _mm512_set1_epi32(0x400)),
                                       _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800)),
                                       rounded);
      }
      scaled =
          _mm512_mask_mov_epi32(scaled, _mm512_cmplt_epi32_mask(scaled, _mm512_set1_epi32(0x400)),
                                _mm512_setzero_si512());
      scaled = _mm512_or_si512(scaled, _mm512_and_si512(result, _mm512_set1_epi32(0x8000)));
      scaled = _mm512_mask_mov_epi32(scaled,
                                     _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400)),
                                     _mm512_setzero_si512());
      result = _mm512_mask_mov_epi32(
          result, _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7c00)), scaled);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7c00));
      auto negative = _mm512_cmpgt_epi32_mask(result, _mm512_set1_epi32(0x7fff));
      auto clamped = _mm512_mask_mov_epi32(_mm512_min_epu32(result, _mm512_set1_epi32(0x3c00)),
                                           nan | negative, _mm512_setzero_si512());
      result = flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? _mm512_mask_mov_epi32(clamped, nan, result)
                                                  : clamped;
    }
    result = _mm512_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    result = _mm512_or_si512(result,
                             _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                              _mm512_set1_epi32(d_shift ? 0xffff : 0xffff0000U)));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}
} // namespace

// Evaluate half sine/cosine under the validated guest policy; D may alias A.
void half_trig_x86_64_v4(bool cosine, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a) {
  if (cosine) {
    run<true>(flags, exec_mask, mode, d, a);
  } else {
    run<false>(flags, exec_mask, mode, d, a);
  }
}
} // namespace rocjitsu::arith
