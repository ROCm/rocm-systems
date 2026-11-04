// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 DPP8/DPP16 source routing and readable-source masks.
// Requires validated controls; destination filtering belongs to register/dpp.cpp.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "register/dpp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Write every source-snapshot word, zeroing unreadable sources.
// Requires validated controls and disjoint source/output storage.
void dpp8_x86_64_v4(uint32_t exec_mask, uint32_t selectors, bool fi, uint32_t *out,
                    const uint32_t *a) {
  auto shifts = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21);
  auto index = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(selectors)), shifts),
                                _mm512_set1_epi32(7));
  index = _mm512_or_si512(index, _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8));
  for (int lane = 0; lane < 32; lane += 16) {
    auto value = _mm512_permutexvar_epi32(index, _mm512_loadu_si512(a + lane));
    if (!fi) {
      auto lane_exec_mask =
          _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(exec_mask >> lane)), index),
                           _mm512_set1_epi32(1));
      value = _mm512_and_si512(value, _mm512_sub_epi32(_mm512_setzero_si512(), lane_exec_mask));
    }
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + lane), value);
  }
}

namespace {
// Map lane positions through the selected DPP16 control; caller handles invalid-source masks.
__m512i dpp16_index(uint32_t ctrl, __m512i position) {
  if (ctrl <= 0xff) {
    auto shift = _mm512_slli_epi32(_mm512_and_si512(position, _mm512_set1_epi32(3)), 1);
    auto pick = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(ctrl)), shift),
                                 _mm512_set1_epi32(3));
    return _mm512_or_si512(_mm512_andnot_si512(_mm512_set1_epi32(3), position), pick);
  }
  if (ctrl <= 0x10f) {
    return _mm512_and_si512(_mm512_add_epi32(position, _mm512_set1_epi32(int(ctrl & 15))),
                            _mm512_set1_epi32(15));
  }
  if (ctrl <= 0x12f) {
    return _mm512_and_si512(_mm512_sub_epi32(position, _mm512_set1_epi32(int(ctrl & 15))),
                            _mm512_set1_epi32(15));
  }
  if (ctrl == 0x140) {
    return _mm512_xor_si512(position, _mm512_set1_epi32(15));
  }
  if (ctrl == 0x141) {
    return _mm512_xor_si512(position, _mm512_set1_epi32(7));
  }
  if (ctrl <= 0x15f) {
    return _mm512_set1_epi32(int(ctrl & 15));
  }
  return _mm512_xor_si512(position, _mm512_set1_epi32(int(ctrl & 15)));
}
} // namespace

// Write every source-snapshot word and return the readable-source mask.
// With FI set, source readability is independent of EXEC.
// Requires validated controls and disjoint source/output storage.
uint32_t dpp16_x86_64_v4(uint32_t exec_mask, uint32_t control, uint32_t valid_row, bool fi,
                         uint32_t *out, const uint32_t *a) {
  auto positions = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  auto index0 = dpp16_index(control, positions);
  auto bits = _mm512_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                16384, 32768);
  uint32_t readable = 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto index = index0;
    auto value = _mm512_permutexvar_epi32(index, _mm512_loadu_si512(a + lane));
    auto valid =
        _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_set1_epi32(int(valid_row)), bits), bits);
    if (!fi) {
      auto lane_exec_mask =
          _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(exec_mask >> lane)), index),
                           _mm512_set1_epi32(1));
      valid &= _mm512_cmpneq_epi32_mask(lane_exec_mask, _mm512_setzero_si512());
    }
    value = _mm512_maskz_mov_epi32(valid, value);
    readable |= uint32_t(valid) << lane;
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + lane), value);
  }
  return readable;
}

} // namespace rocjitsu::arith
