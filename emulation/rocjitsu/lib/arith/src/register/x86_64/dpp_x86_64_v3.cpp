// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 DPP8/DPP16 source routing, encoding validation, and effective EXEC masks.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "register/dpp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Write all 32 permuted source words, zeroing inactive sources unless FI is set.
// Requires AVX2 and validated controls; out must not overlap a.
void dpp8_x86_64_v3(uint32_t exec_mask, uint32_t selectors, bool fi, uint32_t *out,
                    const uint32_t *a) {
  auto shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
  auto index = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(selectors)), shifts),
                                _mm256_set1_epi32(7));
  for (int lane = 0; lane < 32; lane += 8) {
    auto value = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), index);
    if (!fi) {
      auto lane_exec_mask =
          _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)), index),
                           _mm256_set1_epi32(1));
      value = _mm256_and_si256(value, _mm256_sub_epi32(_mm256_setzero_si256(), lane_exec_mask));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + lane), value);
  }
}

namespace {
// Map lane positions through the selected DPP16 control; caller handles invalid-source masks.
__m256i dpp16_index(uint32_t ctrl, __m256i position) {
  if (ctrl <= 0xff) {
    auto shift = _mm256_slli_epi32(_mm256_and_si256(position, _mm256_set1_epi32(3)), 1);
    auto pick = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(ctrl)), shift),
                                 _mm256_set1_epi32(3));
    return _mm256_or_si256(_mm256_andnot_si256(_mm256_set1_epi32(3), position), pick);
  }
  if (ctrl <= 0x10f) {
    return _mm256_and_si256(_mm256_add_epi32(position, _mm256_set1_epi32(int(ctrl & 15))),
                            _mm256_set1_epi32(15));
  }
  if (ctrl <= 0x12f) {
    return _mm256_and_si256(_mm256_sub_epi32(position, _mm256_set1_epi32(int(ctrl & 15))),
                            _mm256_set1_epi32(15));
  }
  if (ctrl == 0x140) {
    return _mm256_xor_si256(position, _mm256_set1_epi32(15));
  }
  if (ctrl == 0x141) {
    return _mm256_xor_si256(position, _mm256_set1_epi32(7));
  }
  if (ctrl <= 0x15f) {
    return _mm256_set1_epi32(int(ctrl & 15));
  }
  return _mm256_xor_si256(position, _mm256_set1_epi32(int(ctrl & 15)));
}
} // namespace

// Write all 32 source words and return EXEC after row/bank/source filtering.
// Requires AVX2 and validated controls; out must not overlap a.
uint32_t dpp16_x86_64_v3(uint32_t exec_mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a) {
  auto positions = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  auto index0 = dpp16_index(control, positions);
  auto index1 = dpp16_index(control, _mm256_add_epi32(positions, _mm256_set1_epi32(8)));
  auto bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  uint32_t readable = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto index = lane & 8 ? index1 : index0;
    int base = lane & 16;
    auto low = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base)), index);
    auto high = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base + 8)), index);
    auto value = _mm256_blendv_epi8(low, high, _mm256_cmpgt_epi32(index, _mm256_set1_epi32(7)));
    auto valid = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_set1_epi32(int(valid_row >> (lane & 8))), bits), bits);
    if (!fi) {
      auto lane_exec_mask =
          _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(exec_mask >> base)), index),
                           _mm256_set1_epi32(1));
      valid = _mm256_and_si256(valid, _mm256_sub_epi32(_mm256_setzero_si256(), lane_exec_mask));
    }
    value = _mm256_and_si256(value, valid);
    readable |= uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(valid))) << lane;
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + lane), value);
  }
  return readable;
}

} // namespace rocjitsu::arith
