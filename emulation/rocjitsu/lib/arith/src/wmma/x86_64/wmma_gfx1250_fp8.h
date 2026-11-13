// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 byte-input WMMA with SIMD widening and shared row FMAs.
#pragma once
#include "common/x86_64/fp8.h"
#include "wmma/gfx1250_fp8.h"
#include "wmma/gfx1250_half.h"
#include "wmma/x86_64/gfx1250_reduction.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Stage every source before publication, preserving whole-register operand overlap.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_kernel(uint64_t flags, uint64_t controls, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr auto form = Packed ? Wmma1250Form::F16 : Wmma1250Form::F32F16;
  alignas(64) float left[16 * K], right[16 * K];
  for (int i = 0; i < 16 * K; i += 8) {
    auto x = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(a[i / 128] + i / 4 % 32));
    auto y = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b[i / 128] + i / 4 % 32));
    _mm256_store_ps(left + i, widen_fp8<Bf8A>(_mm256_cvtepu8_epi32(x)));
    _mm256_store_ps(right + i, widen_fp8<Bf8B>(_mm256_cvtepu8_epi32(y)));
  }
  alignas(64) float rows[16][K], columns[K][16], result[16][16];
  for (int row = 0; row < 16; ++row) {
    for (int k = 0; k < K; ++k) {
      rows[row][k] = left[gfx1250_fp8_index(row, k)];
      columns[k][row] = right[gfx1250_fp8_index(row, k)];
    }
    for (int col = 0; col < 16; ++col) {
      result[row][col] = gfx1250_read_c<form>(c, row, col, controls);
    }
  }
  gfx1250_reduce_rows<K, false>(rows, columns, result);
  gfx1250_write_result<form>(flags, d, result, flags & RJ_ARITH_FP16_OVFL);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
