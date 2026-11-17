// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 24-bit and packed 16-bit integer multiply-add.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_mad.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract the selected narrow factor and sign- or zero-extend it to 32-bit lanes.
template <int Bits, bool Signed> __m512i input(__m512i value, int shift) {
  if constexpr (Bits == 16) {
    value = _mm512_srlv_epi32(value, _mm512_set1_epi32(shift));
  }
  if constexpr (Signed) {
    return _mm512_srai_epi32(_mm512_slli_epi32(value, 32 - Bits), 32 - Bits);
  } else {
    return _mm512_and_si512(value, _mm512_set1_epi32((1 << Bits) - 1));
  }
}

// Return selected even/odd multiply-add lanes clamped to the signed/unsigned 32-bit range.
template <bool Signed, bool Odd> __m512i saturated(__m512i a, __m512i b, __m512i c) {
  if constexpr (Odd) {
    a = _mm512_srli_epi64(a, 32);
    b = _mm512_srli_epi64(b, 32);
    c = _mm512_srli_epi64(c, 32);
  }
  const auto low_word = _mm512_set1_epi64(INT64_C(0xffffffff));
  auto addend = _mm512_and_si512(c, low_word);
  if constexpr (Signed) {
    addend = _mm512_or_si512(addend, _mm512_slli_epi64(_mm512_srai_epi32(c, 31), 32));
  }
  auto product = Signed ? _mm512_mul_epi32(a, b) : _mm512_mul_epu32(a, b);
  auto value = _mm512_add_epi64(product, addend);
  if constexpr (Signed) {
    auto minimum = _mm512_set1_epi64(INT32_MIN), maximum = _mm512_set1_epi64(INT32_MAX);
    return _mm512_min_epi64(_mm512_max_epi64(value, minimum), maximum);
  } else {
    // A 24-bit unsigned product plus a 32-bit addend is below 2^49, so
    // signed 64-bit comparisons suffice for the unsigned saturation test.
    return _mm512_min_epi64(value, low_word);
  }
}

} // namespace

// Multiply-add narrow integer factors. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed>
void integer_mad_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c) {
  const int sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sb = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto x =
        input<Bits, Signed>(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane)), sa);
    auto y =
        input<Bits, Signed>(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane)), sb);
    auto z = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
    __m512i result;
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto even = saturated<Signed, false>(x, y, z), odd = saturated<Signed, true>(x, y, z);
      result = _mm512_or_si512(_mm512_and_si512(even, _mm512_set1_epi64(INT64_C(0xffffffff))),
                               _mm512_slli_epi64(odd, 32));
    } else {
      result = _mm512_add_epi32(_mm512_mullo_epi32(x, y), z);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer_mad_x86_64_v4<16, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v4<16, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v4<24, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v4<24, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
