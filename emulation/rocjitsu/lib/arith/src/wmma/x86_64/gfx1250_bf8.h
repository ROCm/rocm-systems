// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Widen the low two E5M2 bytes per VGPR lane as high binary16 bytes. Callers
// recover the matrix NaN payload before using or publishing the value.
#pragma once
#include "wmma/x86_64/gfx1250_accumulator.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
inline void gfx1250_widen_bf8_pair(Gfx1250PackedWords packed, Gfx1250PackedFloat &first,
                                   Gfx1250PackedFloat &second) {
#if defined(__AVX512F__)
  auto pair = _mm512_cvtepi32_epi16(bit_cast<__m512i>(packed));
  first = _mm512_cvtph_ps(_mm256_slli_epi16(pair, 8));
  second = _mm512_cvtph_ps(_mm256_and_si256(pair, _mm256_set1_epi16(static_cast<int16_t>(0xff00))));
#else
  auto raw = bit_cast<__m256i>(packed & 0xffffu);
  auto pair = _mm_packus_epi32(_mm256_castsi256_si128(raw), _mm256_extracti128_si256(raw, 1));
  first = _mm256_cvtph_ps(_mm_slli_epi16(pair, 8));
  second = _mm256_cvtph_ps(_mm_and_si128(pair, _mm_set1_epi16(static_cast<int16_t>(0xff00))));
#endif
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
