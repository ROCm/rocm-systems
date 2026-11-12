// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 16-bit multiply-add and three-input integer selections.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer16_ternary.h"
#include "integer/x86_64/packed_integer_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Extract the selected halfword and sign- or zero-extend it to 32-bit lanes.
template <bool Signed, bool High> __m256i widen(__m256i value) {
  if constexpr (Signed) {
    return _mm256_srai_epi32(High ? value : _mm256_slli_epi32(value, 16), 16);
  } else {
    return High ? _mm256_srli_epi32(value, 16) : _mm256_and_si256(value, _mm256_set1_epi32(65535));
  }
}

// Return halfword multiply-add results clamped to signed/unsigned 16-bit range.
template <bool Signed, bool High> __m256i saturated(__m256i a, __m256i b, __m256i c) {
  // Even the largest unsigned result, 65535 * 65535 + 65535, fits uint32_t.
  auto value = _mm256_add_epi32(_mm256_mullo_epi32(widen<Signed, High>(a), widen<Signed, High>(b)),
                                widen<Signed, High>(c));
  if constexpr (Signed) {
    return _mm256_min_epi32(_mm256_max_epi32(value, _mm256_set1_epi32(-32768)),
                            _mm256_set1_epi32(32767));
  } else {
    return _mm256_min_epu32(value, _mm256_set1_epi32(65535));
  }
}

} // namespace

// Evaluate three-input 16-bit integer arithmetic. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Signed, Integer16Ternary Op, bool Packed>
void integer16_ternary_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b, const uint32_t *c) {
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  if constexpr (!Packed) {
    mode = (mode & RJ_ARITH_ALU_HIGH_A ? RJ_ARITH_PK_LO_A_HIGH : 0) |
           (mode & RJ_ARITH_ALU_HIGH_B ? RJ_ARITH_PK_LO_B_HIGH : 0) |
           (mode & RJ_ARITH_ALU_HIGH_C ? RJ_ARITH_PK_LO_C_HIGH : 0) |
           (mode & RJ_ARITH_ALU_CLAMP ? RJ_ARITH_PK_CLAMP : 0);
  }
  auto sa = packed_integer_selection(mode & RJ_ARITH_PK_LO_A_HIGH, mode & RJ_ARITH_PK_HI_A_LOW);
  auto sb = packed_integer_selection(mode & RJ_ARITH_PK_LO_B_HIGH, mode & RJ_ARITH_PK_HI_B_LOW);
  auto sc = packed_integer_selection(mode & RJ_ARITH_PK_LO_C_HIGH, mode & RJ_ARITH_PK_HI_C_LOW);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), sa);
    auto y =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), sb);
    auto z =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), sc);
    __m256i result;
    if constexpr (Op == Integer16Ternary::Mad) {
      if (mode & RJ_ARITH_PK_CLAMP) {
        auto low = saturated<Signed, false>(x, y, z), high = saturated<Signed, true>(x, y, z);
        result = _mm256_or_si256(_mm256_and_si256(low, _mm256_set1_epi32(65535)),
                                 _mm256_slli_epi32(high, 16));
      } else {
        result = _mm256_add_epi16(_mm256_mullo_epi16(x, y), z);
      }
    } else if constexpr (Op == Integer16Ternary::Min) {
      result = Signed ? _mm256_min_epi16(_mm256_min_epi16(x, y), z)
                      : _mm256_min_epu16(_mm256_min_epu16(x, y), z);
    } else if constexpr (Op == Integer16Ternary::Max) {
      result = Signed ? _mm256_max_epi16(_mm256_max_epi16(x, y), z)
                      : _mm256_max_epu16(_mm256_max_epu16(x, y), z);
    } else {
      auto lower = Signed ? _mm256_min_epi16(x, y) : _mm256_min_epu16(x, y);
      auto upper = Signed ? _mm256_max_epi16(x, y) : _mm256_max_epu16(x, y);
      auto limited = Signed ? _mm256_min_epi16(upper, z) : _mm256_min_epu16(upper, z);
      result = Signed ? _mm256_max_epi16(lower, limited) : _mm256_max_epu16(lower, limited);
    }
    if constexpr (!Packed) {
      auto output_mask = _mm256_set1_epi32(int(65535U << d_shift));
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      auto selected = _mm256_sllv_epi32(result, _mm256_set1_epi32(d_shift));
      result = _mm256_or_si256(_mm256_and_si256(selected, output_mask),
                               _mm256_andnot_si256(output_mask, original));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void integer16_ternary_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                                const uint32_t *, const uint32_t *);

template void integer16_ternary_x86_64_v3<false, Integer16Ternary::Mad, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<true, Integer16Ternary::Mad, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<false, Integer16Ternary::Min, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<true, Integer16Ternary::Min, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<false, Integer16Ternary::Max, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<true, Integer16Ternary::Max, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);

template void integer16_ternary_x86_64_v3<false, Integer16Ternary::Median, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);
template void integer16_ternary_x86_64_v3<true, Integer16Ternary::Median, false>(
    uint32_t, uint32_t, uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
