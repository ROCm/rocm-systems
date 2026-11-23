// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local packed-register vector loads for sparse WMMA.
#pragma once

#include "common/bits.h"
#include <cstring>
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86 {
#if defined(__AVX512F__)
constexpr int Width = 16;
#else
constexpr int Width = 8;
#endif
using U32 [[gnu::vector_size(Width * 4)]] = uint32_t;
using F32 [[gnu::vector_size(Width * 4)]] = float;

inline U32 load(const uint32_t *p) {
  U32 x;
  std::memcpy(&x, p, sizeof(x));
  return x;
}

// Each 32-bit lane holds a metadata nibble. Its upper bytes read table[0], which is zero.
template <int Stride> inline U32 pair_offset(U32 nibbles) {
  static_assert(Stride > 0 && 5 * Stride <= 255);
  // Valid ascending metadata pairs map to offsets 0 through 5 * Stride.
  static constexpr uint8_t table[16] = {
      0, 0, 0, 0, 0, 0, 0, 0, Stride, 2 * Stride, 0, 0, 3 * Stride, 4 * Stride, 5 * Stride, 0};
  const __m128i entries = _mm_loadu_si128(reinterpret_cast<const __m128i *>(table));
#if defined(__AVX512F__)
  return bit_cast<U32>(
      _mm512_shuffle_epi8(_mm512_broadcast_i32x4(entries), bit_cast<__m512i>(nibbles)));
#else
  return bit_cast<U32>(
      _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(entries), bit_cast<__m256i>(nibbles)));
#endif
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::sparse_x86
