// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 16-bit integer add/subtract, multiply, shifts, and min/max.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer16.h"
#include "integer/x86_64/packed_integer_x86_64_v4.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate 16-bit integer arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Integer16 Op, bool Signed, bool Packed>
void integer16_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b) {
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  if constexpr (!Packed) {
    mode = (mode & RJ_ARITH_ALU_HIGH_A ? RJ_ARITH_PK_LO_A_HIGH : 0) |
           (mode & RJ_ARITH_ALU_HIGH_B ? RJ_ARITH_PK_LO_B_HIGH : 0) |
           (mode & RJ_ARITH_ALU_CLAMP ? RJ_ARITH_PK_CLAMP : 0);
  }
  auto sa = packed_integer_selection(mode & RJ_ARITH_PK_LO_A_HIGH, mode & RJ_ARITH_PK_HI_A_LOW);
  auto sb = packed_integer_selection(mode & RJ_ARITH_PK_LO_B_HIGH, mode & RJ_ARITH_PK_HI_B_LOW);
  for (int lane = 0; lane < 32; lane += 16) {
    auto x =
        _mm512_shuffle_epi8(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane)), sa);
    auto y =
        _mm512_shuffle_epi8(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane)), sb);
    __m512i result;
    if constexpr (Op == Integer16::Add) {
      result = mode & RJ_ARITH_PK_CLAMP
                   ? (Signed ? _mm512_adds_epi16(x, y) : _mm512_adds_epu16(x, y))
                   : _mm512_add_epi16(x, y);
    }
    if constexpr (Op == Integer16::Sub) {
      result = mode & RJ_ARITH_PK_CLAMP
                   ? (Signed ? _mm512_subs_epi16(x, y) : _mm512_subs_epu16(x, y))
                   : _mm512_sub_epi16(x, y);
    }
    if constexpr (Op == Integer16::Min) {
      result = Signed ? _mm512_min_epi16(x, y) : _mm512_min_epu16(x, y);
    }
    if constexpr (Op == Integer16::Max) {
      result = Signed ? _mm512_max_epi16(x, y) : _mm512_max_epu16(x, y);
    }
    if constexpr (Op == Integer16::Mul || Op == Integer16::MulClamp) {
      result = _mm512_mullo_epi16(x, y);
      if constexpr (Op == Integer16::MulClamp) {
        if (mode & RJ_ARITH_PK_CLAMP) {
          auto overflow =
              _mm512_cmpneq_epi16_mask(_mm512_mulhi_epu16(x, y), _mm512_setzero_si512());
          result = _mm512_mask_mov_epi16(result, overflow, _mm512_set1_epi16(-1));
        }
      }
    }
    if constexpr (Op == Integer16::ShiftLeft || Op == Integer16::ShiftRight) {
      auto count = _mm512_and_si512(x, _mm512_set1_epi16(15));
      if constexpr (Op == Integer16::ShiftLeft) {
        result = _mm512_sllv_epi16(y, count);
      } else if constexpr (Signed) {
        result = _mm512_srav_epi16(y, count);
      } else {
        result = _mm512_srlv_epi16(y, count);
      }
    }
    if constexpr (!Packed) {
      auto output_mask = _mm512_set1_epi32(int(65535U << d_shift));
      auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      auto selected = _mm512_sllv_epi32(result, _mm512_set1_epi32(d_shift));
      result = _mm512_or_si512(_mm512_and_si512(selected, output_mask),
                               _mm512_andnot_si512(output_mask, original));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer16_x86_64_v4<Integer16::Add, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Sub, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Add, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Sub, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Min, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Max, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Min, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Max, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Mul, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);

template void integer16_x86_64_v4<Integer16::ShiftLeft, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::ShiftRight, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::ShiftRight, true>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);

template void integer16_x86_64_v4<Integer16::Add, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Sub, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Add, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Sub, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Min, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Max, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Min, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Max, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::Mul, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::ShiftLeft, false, false>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);
template void integer16_x86_64_v4<Integer16::ShiftRight, false, false>(uint32_t, uint32_t,
                                                                       uint32_t *, const uint32_t *,
                                                                       const uint32_t *);
template void integer16_x86_64_v4<Integer16::ShiftRight, true, false>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);

template void integer16_x86_64_v4<Integer16::MulClamp, false>(uint32_t, uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *);
template void integer16_x86_64_v4<Integer16::MulClamp, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
} // namespace rocjitsu::arith
