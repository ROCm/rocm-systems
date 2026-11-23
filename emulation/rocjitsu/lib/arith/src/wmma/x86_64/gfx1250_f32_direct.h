// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct VGPR reduction for gfx1250 FP32 K4 on AVX2 and AVX-512.
#pragma once
#include "common/compiler.h"
#include "wmma/x86_64/gfx1250_accumulator.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

inline bool gfx1250_f32_all_finite(const uint32_t *const *src) {
  const auto exponent = _mm256_set1_epi32(0x7f800000);
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src[reg] + lane));
      auto special = _mm256_cmpeq_epi32(_mm256_and_si256(x, exponent), exponent);
      if (_mm256_movemask_epi8(special) != 0) {
        return false;
      }
    }
  }
  return true;
}

// Read B and C in their native column order and broadcast A directly from its two VGPRs.
// Stage D in register order so every source is consumed before the first destination store.
[[gnu::noinline]] inline void gfx1250_f32_direct(uint64_t controls, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  using Words = Gfx1250PackedWords;
  using Values = Gfx1250PackedFloat;
#if defined(__AVX512F__)
  auto fma = [](float x, Values y, Values z) { return _mm512_fmadd_ps(_mm512_set1_ps(x), y, z); };
#else
  auto fma = [](float x, Values y, Values z) { return _mm256_fmadd_ps(_mm256_set1_ps(x), y, z); };
#endif
  constexpr int width = sizeof(Words) / sizeof(uint32_t);
  alignas(64) uint32_t result[8][32];
  const uint32_t keep = (controls & RJ_ARITH_WMMA_ABS_C) ? 0x7fffffffU : 0xffffffffU;
  const uint32_t negate = (controls & RJ_ARITH_WMMA_NEG_C) ? 0x80000000U : 0;
  for (int col = 0; col < 16; col += width) {
    Values right[4];
    for (int k = 0; k < 4; ++k) {
      std::memcpy(&right[k], b[k % 2] + 16 * (k / 2) + col, sizeof(Values));
    }
    for (int row = 0; row < 16; ++row) {
      const int reg = row % 8;
      const int lane = 16 * (row / 8) + col;
      Words bits;
      std::memcpy(&bits, c[reg] + lane, sizeof(bits));
      auto acc = bit_cast<Values>((bits & keep) ^ negate);
      RJ_ARITH_UNROLL(4)
      for (int k = 0; k < 4; ++k) {
        acc = fma(as_float(a[k % 2][16 * (k / 2) + row]), right[k], acc);
      }
      std::memcpy(result[reg] + lane, &acc, sizeof(acc));
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    std::memcpy(d[reg], result[reg], sizeof(result[reg]));
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
