// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX-512 sine/cosine polynomial for FP32 and widened FP16 inputs.
// The coefficient data and phase mapping are inherited; lookup stays in vector registers.
#pragma once

#include "common/isa.h"
#include "math/trig_coefficients.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Select an existing coefficient from two sixteen-element register tables.
inline __m512i trig_coefficient(const int32_t *row, __m512i column) {
  return _mm512_permutex2var_epi32(_mm512_loadu_si512(row), column, _mm512_loadu_si512(row + 16));
}

// Evaluate the existing AVX2 reduced polynomial at AVX-512 width, before half rounding.
template <bool Cosine> inline __m512 trig_value(__m512i bits) {
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
  auto c0 = _mm512_cvtepi32_ps(trig_coefficient(trig::coefficients[0], column));
  auto c1 = _mm512_cvtepi32_ps(trig_coefficient(trig::coefficients[1], column));
  auto c2 = _mm512_cvtepi32_ps(trig_coefficient(trig::coefficients[2], column));
  auto c3 = _mm512_cvtepi32_ps(trig_coefficient(trig::coefficients[3], column));
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

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
