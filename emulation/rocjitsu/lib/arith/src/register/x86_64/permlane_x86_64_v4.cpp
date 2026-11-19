// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 lane permutations across rows, half-waves, and whole waves.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "register/permlane.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Exchange adjacent rows with independent destination masks. Load each pair
// before storing so the two operands may refer to the same VGPR.
void swap_rows_x86_64_v4(int lanes, int stride, uint64_t exec_mask, uint32_t *d, uint32_t *a) {
  for (int base = 0; base + stride < lanes; base += 2 * stride) {
    for (int i = 0; i < stride; i += 16) {
      int low = base + i, high = low + stride;
      auto old_a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + low));
      auto old_d = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + high));
      _mm512_mask_storeu_epi32(a + low, __mmask16(exec_mask >> low), old_d);
      _mm512_mask_storeu_epi32(d + high, __mmask16(exec_mask >> high), old_a);
    }
  }
}

// Route within normalized groups, retaining the original lane at invalid boundaries.
// Stage the complete wave before publishing aliases.
void group_permutation_x86_64_v4(GroupPermutation op, int lanes, uint64_t exec_mask, uint32_t *d,
                                 const uint32_t *a, uint32_t selector, uint32_t width) {
  uint32_t result[64];
  auto mask = _mm512_set1_epi32(int(width - 1));
  auto selected = _mm512_set1_epi32(int(selector));
  for (int lane = 0; lane < lanes; lane += 16) {
    auto index =
        _mm512_add_epi32(_mm512_set1_epi32(lane),
                         _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    auto offset = _mm512_and_si512(index, mask);
    auto address = index;
    switch (op) {
    case GroupPermutation::Bcast:
      address = _mm512_or_si512(_mm512_andnot_si512(mask, index), _mm512_and_si512(selected, mask));
      break;
    case GroupPermutation::Up:
      address =
          _mm512_mask_sub_epi32(index, _mm512_cmple_epu32_mask(selected, offset), index, selected);
      break;
    case GroupPermutation::Down: {
      auto remaining = _mm512_sub_epi32(_mm512_set1_epi32(int(width)), offset);
      address = _mm512_mask_add_epi32(index, _mm512_cmplt_epu32_mask(selected, remaining), index,
                                      selected);
      break;
    }
    case GroupPermutation::Xor: {
      auto source = _mm512_xor_si512(index, selected);
      auto upper =
          _mm512_add_epi32(_mm512_andnot_si512(mask, index), _mm512_set1_epi32(int(width)));
      if (selector < uint32_t(lanes)) {
        address = _mm512_mask_mov_epi32(index, _mm512_cmplt_epu32_mask(source, upper), source);
      }
      break;
    }
    }
    auto value = _mm512_i32gather_epi32(address, a, 4);
    _mm512_storeu_si512(result + lane, value);
  }
  for (int lane = 0; lane < lanes; lane += 16) {
    auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result + lane));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

// Permute source lanes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Cross, bool Var>
void permlane_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi) {
  __m512i results[2];
  auto ones = _mm512_set1_epi32(1);
  auto bits = _mm512_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                16384, 32768);
  auto shifts = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
  for (int lane = 0; lane < 32; lane += 16) {
    __m512i index;
    if constexpr (Var) {
      index = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    } else {
      auto packed =
          _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(int(lo)), _mm512_set1_epi32(int(hi)));
      index = _mm512_srlv_epi32(packed, shifts);
    }
    index = _mm512_and_si512(index, _mm512_set1_epi32(15));
    int base = (lane & 16) ^ (Cross ? 16 : 0);
    auto value = _mm512_permutexvar_epi32(index, _mm512_loadu_si512(a + base));
    if (!(mode & RJ_ARITH_PERMLANE_FI)) {
      auto lane_exec_mask = _mm512_and_si512(
          _mm512_srlv_epi32(_mm512_set1_epi32(int(exec_mask >> base)), index), ones);
      auto fallback = mode & RJ_ARITH_PERMLANE_BOUND_CTRL
                          ? _mm512_setzero_si512()
                          : _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      value = _mm512_mask_blend_epi32(
          _mm512_cmpneq_epi32_mask(lane_exec_mask, _mm512_setzero_si512()), fallback, value);
    }
    results[lane / 16] = value;
  }
  // Defer every store until both source rows and all indices have been read.
  for (int lane = 0; lane < 32; lane += 16) {
    (void)bits;
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), results[lane / 16]);
  }
}

template void permlane_x86_64_v4<false, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<false, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<true, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v4<true, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *, uint32_t, uint32_t);

// Stage the complete wave before masked stores, including overlapping source ranges.
void permlane_index_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                              uint32_t group_width) {
  __m512i result[64 / 16];
  auto mask = _mm512_set1_epi32(int(group_width - 1));
  for (int lane = 0; lane < lanes; lane += 16) {
    auto index =
        _mm512_add_epi32(_mm512_set1_epi32(lane),
                         _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    auto source = _mm512_and_si512(_mm512_loadu_si512(a + lane),
                                   _mm512_set1_epi32(int(63 & (group_width - 1))));
    result[lane / 16] =
        _mm512_slli_epi32(_mm512_or_si512(_mm512_andnot_si512(mask, index), source), 2);
  }
  for (int lane = 0; lane < lanes; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result[lane / 16]);
  }
}
} // namespace rocjitsu::arith
