// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 lane permutations across rows, half-waves, and whole waves.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "register/permlane.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Exchange adjacent rows with independent destination masks. Load each pair
// before storing so the two operands may refer to the same VGPR.
void swap_rows_x86_64_v3(int lanes, int stride, uint64_t exec_mask, uint32_t *d, uint32_t *a) {
  for (int base = 0; base + stride < lanes; base += 2 * stride) {
    for (int i = 0; i < stride; i += 8) {
      int low = base + i, high = low + stride;
      auto old_a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + low));
      auto old_d = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + high));
      auto shifts = _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24);
      auto a_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> low))), shifts);
      auto d_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> high))), shifts);
      _mm256_maskstore_epi32(reinterpret_cast<int *>(a + low), a_mask, old_d);
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d + high), d_mask, old_a);
    }
  }
}

// Route within normalized groups, retaining the original lane at invalid boundaries.
// Stage the complete wave before publishing aliases.
void group_permutation_x86_64_v3(GroupPermutation op, int lanes, uint64_t exec_mask, uint32_t *d,
                                 const uint32_t *a, uint32_t selector, uint32_t width) {
  uint32_t result[64];
  auto mask = _mm256_set1_epi32(int(width - 1));
  auto selected = _mm256_set1_epi32(int(selector));
  for (int lane = 0; lane < lanes; lane += 8) {
    auto index =
        _mm256_add_epi32(_mm256_set1_epi32(lane), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    auto offset = _mm256_and_si256(index, mask);
    auto address = index;
    switch (op) {
    case GroupPermutation::Bcast:
      address = _mm256_or_si256(_mm256_andnot_si256(mask, index), _mm256_and_si256(selected, mask));
      break;
    case GroupPermutation::Up: {
      auto valid = _mm256_cmpeq_epi32(_mm256_min_epu32(selected, offset), selected);
      address = _mm256_blendv_epi8(index, _mm256_sub_epi32(index, selected), valid);
      break;
    }
    case GroupPermutation::Down: {
      auto remaining = _mm256_sub_epi32(_mm256_set1_epi32(int(width)), offset);
      auto valid = _mm256_cmpgt_epi32(remaining, selected);
      // Signed comparison is valid only after ruling out selectors above the wave size.
      if (selector < width) {
        address = _mm256_blendv_epi8(index, _mm256_add_epi32(index, selected), valid);
      }
      break;
    }
    case GroupPermutation::Xor: {
      auto source = _mm256_xor_si256(index, selected);
      auto upper =
          _mm256_add_epi32(_mm256_andnot_si256(mask, index), _mm256_set1_epi32(int(width)));
      if (selector < uint32_t(lanes)) {
        address = _mm256_blendv_epi8(index, source, _mm256_cmpgt_epi32(upper, source));
      }
      break;
    }
    }
    auto value = _mm256_i32gather_epi32(reinterpret_cast<const int *>(a), address, 4);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), value);
  }
  for (int lane = 0; lane < lanes; lane += 8) {
    auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

// Permute source lanes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Cross, bool Var>
void permlane_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi) {
  __m256i results[4];
  auto ones = _mm256_set1_epi32(1);
  auto bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  auto shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  for (int lane = 0; lane < 32; lane += 8) {
    __m256i index;
    if constexpr (Var) {
      index = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    } else {
      auto packed = _mm256_set1_epi32(int(lane & 8 ? hi : lo));
      index = _mm256_srlv_epi32(packed, shifts);
    }
    index = _mm256_and_si256(index, _mm256_set1_epi32(15));
    int base = (lane & 16) ^ (Cross ? 16 : 0);
    auto low = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base)), index);
    auto high = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base + 8)), index);
    auto value = _mm256_blendv_epi8(low, high, _mm256_cmpgt_epi32(index, _mm256_set1_epi32(7)));
    if (!(mode & RJ_ARITH_PERMLANE_FI)) {
      auto lane_exec_mask = _mm256_and_si256(
          _mm256_srlv_epi32(_mm256_set1_epi32(int(exec_mask >> base)), index), ones);
      auto fallback = mode & RJ_ARITH_PERMLANE_BOUND_CTRL
                          ? _mm256_setzero_si256()
                          : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      value = _mm256_blendv_epi8(fallback, value,
                                 _mm256_sub_epi32(_mm256_setzero_si256(), lane_exec_mask));
    }
    results[lane / 8] = value;
  }
  // Defer every store until both source rows and all indices have been read.
  for (int lane = 0; lane < 32; lane += 8) {
    auto enabled =
        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(int(exec_mask >> lane)), bits), bits);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), enabled, results[lane / 8]);
  }
}

template void permlane_x86_64_v3<false, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<false, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<true, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<true, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *, uint32_t, uint32_t);

// Evaluate permlane64 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void permlane64_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 8) {
    // Read both sides of the exchange before either store, including D == A.
    auto lower = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto upper = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane + 32));
    auto low_active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> lane)),
                                        _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    auto high_active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> (lane + 32))),
                                         _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), low_active, upper);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane + 32), high_active, lower);
  }
}

// Stage the complete wave before masked stores, including overlapping source ranges.
void permlane_index_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                              uint32_t group_width) {
  __m256i result[64 / 8];
  auto mask = _mm256_set1_epi32(int(group_width - 1));
  for (int lane = 0; lane < lanes; lane += 8) {
    auto index =
        _mm256_add_epi32(_mm256_set1_epi32(lane), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    auto source = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                                   _mm256_set1_epi32(int(63 & (group_width - 1))));
    result[lane / 8] =
        _mm256_slli_epi32(_mm256_or_si256(_mm256_andnot_si256(mask, index), source), 2);
  }
  for (int lane = 0; lane < lanes; lane += 8) {
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result[lane / 8]);
  }
}
} // namespace rocjitsu::arith
