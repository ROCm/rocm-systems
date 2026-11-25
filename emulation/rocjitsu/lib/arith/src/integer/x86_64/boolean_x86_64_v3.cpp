// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 wordwise boolean operations and the stochastic-rounding LFSR step.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/boolean.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate wordwise boolean operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Boolean Op, bool Half>
void boolean_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sb = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination_mask =
      _mm256_set1_epi32(int(uint32_t(65535) << (mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0)));
  for (int lane = 0; lane < 32; lane += 8) {
    __m256i x;
    x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    if constexpr (Half) {
      x = _mm256_srl_epi32(x, sa);
    }
    __m256i y;
    if constexpr (Op != Boolean::Not && Op != Boolean::Prng) {
      y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      if constexpr (Half) {
        y = _mm256_srl_epi32(y, sb);
      }
    }
    __m256i result;
    if constexpr (Op == Boolean::And) {
      result = _mm256_and_si256(x, y);
    }
    if constexpr (Op == Boolean::Or) {
      result = _mm256_or_si256(x, y);
    }
    if constexpr (Op == Boolean::Xor) {
      result = _mm256_xor_si256(x, y);
    }
    if constexpr (Op == Boolean::Xnor) {
      result = _mm256_xor_si256(_mm256_xor_si256(x, y), _mm256_set1_epi32(-1));
    }
    if constexpr (Op == Boolean::Not) {
      result = _mm256_xor_si256(x, _mm256_set1_epi32(-1));
    }
    if constexpr (Op == Boolean::Prng) {
      auto feedback = _mm256_and_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32(0xc5));
      result = _mm256_xor_si256(_mm256_slli_epi32(x, 1), feedback);
    }
    if constexpr (Half) {
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      auto selected = _mm256_sll_epi32(result, sd);
      result = _mm256_or_si256(_mm256_and_si256(destination_mask, selected),
                               _mm256_andnot_si256(destination_mask, original));
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void boolean_x86_64_v3<Boolean::And, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v3<Boolean::Or, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v3<Boolean::Xor, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v3<Boolean::Not, false>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                     const uint32_t *a, const uint32_t *b);
template void boolean_x86_64_v3<Boolean::And, true>(uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                                    const uint32_t *a, const uint32_t *b);

template void boolean_x86_64_v3<Boolean::Xnor, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);

template void boolean_x86_64_v3<Boolean::Prng, false>(uint32_t, uint32_t, uint32_t *,
                                                      const uint32_t *, const uint32_t *);

// Apply a runtime truth table using wordwise selection, retaining the fast path for all tables.
template <bool Half>
void bitop3_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b, const uint32_t *c) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sb = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0);
  auto sc = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination_mask = _mm256_set1_epi32(int(0xffffU << (mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0)));
  auto bit = [mode](int i) { return _mm256_set1_epi32(-int((mode >> i) & 1)); };
  auto choose = [](__m256i mask, __m256i yes, __m256i no) {
    return _mm256_or_si256(_mm256_and_si256(mask, yes), _mm256_andnot_si256(mask, no));
  };
  for (int lane = 0; lane < 32; lane += 8) {
    auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto cv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    if constexpr (Half) {
      av = _mm256_srl_epi32(av, sa);
      bv = _mm256_srl_epi32(bv, sb);
      cv = _mm256_srl_epi32(cv, sc);
    }
    auto result = choose(av, choose(bv, choose(cv, bit(7), bit(6)), choose(cv, bit(5), bit(4))),
                         choose(bv, choose(cv, bit(3), bit(2)), choose(cv, bit(1), bit(0))));
    if constexpr (Half) {
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      result = choose(destination_mask, _mm256_sll_epi32(result, sd), original);
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void bitop3_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                      const uint32_t *, const uint32_t *);
template void bitop3_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                     const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
