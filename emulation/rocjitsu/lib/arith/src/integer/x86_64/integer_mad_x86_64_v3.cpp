// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 24-bit and packed 16-bit integer multiply-add.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_mad.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract the selected narrow factor and sign- or zero-extend it to 32-bit lanes.
template <int Bits, bool Signed> __m256i input(__m256i value, int shift) {
  if constexpr (Bits == 16) {
    value = _mm256_srlv_epi32(value, _mm256_set1_epi32(shift));
  }
  if constexpr (Signed) {
    return _mm256_srai_epi32(_mm256_slli_epi32(value, 32 - Bits), 32 - Bits);
  } else {
    return _mm256_and_si256(value, _mm256_set1_epi32((1 << Bits) - 1));
  }
}

// Return selected even/odd multiply-add lanes clamped to the signed/unsigned 32-bit range.
template <bool Signed, bool Odd> __m256i saturated(__m256i a, __m256i b, __m256i c) {
  if constexpr (Odd) {
    a = _mm256_srli_epi64(a, 32);
    b = _mm256_srli_epi64(b, 32);
    c = _mm256_srli_epi64(c, 32);
  }
  const auto low_word = _mm256_set1_epi64x(INT64_C(0xffffffff));
  auto addend = _mm256_and_si256(c, low_word);
  if constexpr (Signed) {
    addend = _mm256_or_si256(addend, _mm256_slli_epi64(_mm256_srai_epi32(c, 31), 32));
  }
  auto product = Signed ? _mm256_mul_epi32(a, b) : _mm256_mul_epu32(a, b);
  auto value = _mm256_add_epi64(product, addend);
  if constexpr (Signed) {
    auto minimum = _mm256_set1_epi64x(INT32_MIN), maximum = _mm256_set1_epi64x(INT32_MAX);
    value = _mm256_blendv_epi8(value, minimum, _mm256_cmpgt_epi64(minimum, value));
    return _mm256_blendv_epi8(value, maximum, _mm256_cmpgt_epi64(value, maximum));
  } else {
    // A 24-bit unsigned product plus a 32-bit addend is below 2^49, so
    // signed 64-bit comparisons suffice for the unsigned saturation test.
    return _mm256_blendv_epi8(value, low_word, _mm256_cmpgt_epi64(value, low_word));
  }
}

} // namespace

// Multiply-add narrow integer factors. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed>
void integer_mad_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c) {
  const int sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sb = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x =
        input<Bits, Signed>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), sa);
    auto y =
        input<Bits, Signed>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), sb);
    auto z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    __m256i result;
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto even = saturated<Signed, false>(x, y, z), odd = saturated<Signed, true>(x, y, z);
      result = _mm256_or_si256(_mm256_and_si256(even, _mm256_set1_epi64x(INT64_C(0xffffffff))),
                               _mm256_slli_epi64(odd, 32));
    } else {
      result = _mm256_add_epi32(_mm256_mullo_epi32(x, y), z);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void integer_mad_x86_64_v3<16, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v3<16, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v3<24, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, const uint32_t *);
template void integer_mad_x86_64_v3<24, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
