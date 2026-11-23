// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed U64 wrapping/saturating arithmetic, staged before alias-safe masked stores.
#pragma once
#include "common/validation.h"
#include "integer/packed_u64.h"
#include "rj_arith/detail/rj_arith_common.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
#if defined(__AVX512F__)
using PackedWord = __m512i;
constexpr int kPackedLanes = 8;
// Widen one VGPR plane and combine low/high planes into native U64 lanes.
inline PackedWord packed_words(const uint32_t *p) {
  return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
}
inline PackedWord packed_pair(const uint32_t *const *p, int lane) {
  return _mm512_or_si512(packed_words(p[0] + lane),
                         _mm512_slli_epi64(packed_words(p[1] + lane), 32));
}
// Add or subtract with optional unsigned saturation; negate applies after an unclamped sum.
inline PackedWord packed_sum(PackedWord x, PackedWord y, bool difference, bool negate, bool clamp) {
  auto value = difference ? _mm512_sub_epi64(x, y) : _mm512_add_epi64(x, y);
  if (clamp) {
    if (difference) {
      return _mm512_maskz_mov_epi64(_mm512_cmp_epu64_mask(x, y, _MM_CMPINT_NLT), value);
    }
    if (negate) {
      return _mm512_setzero_si512();
    }
    return _mm512_mask_mov_epi64(value, _mm512_cmp_epu64_mask(value, x, _MM_CMPINT_LT),
                                 _mm512_set1_epi64(-1));
  }
  return negate ? _mm512_sub_epi64(_mm512_setzero_si512(), value) : value;
}
// Store split VGPR planes in local staging, without publishing any destination yet.
inline void packed_stage(uint32_t *lo, uint32_t *hi, PackedWord value) {
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(lo), _mm512_cvtepi64_epi32(value));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(hi),
                      _mm512_cvtepi64_epi32(_mm512_srli_epi64(value, 32)));
}
#elif defined(__AVX2__)
using PackedWord = __m256i;
constexpr int kPackedLanes = 4;
// Widen one VGPR plane and combine low/high planes into native U64 lanes.
inline PackedWord packed_words(const uint32_t *p) {
  return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
}
inline PackedWord packed_pair(const uint32_t *const *p, int lane) {
  return _mm256_or_si256(packed_words(p[0] + lane),
                         _mm256_slli_epi64(packed_words(p[1] + lane), 32));
}
// Add or subtract with optional unsigned saturation; negate applies after an unclamped sum.
inline PackedWord packed_sum(PackedWord x, PackedWord y, bool difference, bool negate, bool clamp) {
  auto value = difference ? _mm256_sub_epi64(x, y) : _mm256_add_epi64(x, y);
  if (clamp) {
    const auto bias = _mm256_set1_epi64x(INT64_MIN);
    if (difference) {
      return _mm256_andnot_si256(
          _mm256_cmpgt_epi64(_mm256_xor_si256(y, bias), _mm256_xor_si256(x, bias)), value);
    }
    if (negate) {
      return _mm256_setzero_si256();
    }
    return _mm256_or_si256(
        value, _mm256_cmpgt_epi64(_mm256_xor_si256(x, bias), _mm256_xor_si256(value, bias)));
  }
  return negate ? _mm256_sub_epi64(_mm256_setzero_si256(), value) : value;
}
// Store split VGPR planes in local staging, without publishing any destination yet.
inline void packed_stage(uint32_t *lo, uint32_t *hi, PackedWord value) {
  auto planes = _mm256_permutevar8x32_epi32(value, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(lo), _mm256_castsi256_si128(planes));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(hi), _mm256_extracti128_si256(planes, 1));
}
#endif
// Evaluate both packed elements before any store, including overlapping destination registers.
template <PackedU64 Op>
void packed_u64_kernel(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  uint32_t staged[4][32];
  for (int element = 0; element < 2; ++element) {
    const bool negative_a = mode & (RJ_ARITH_PK_NEG_LO_A << (element * 3));
    const bool negative_b =
        bool(mode & (RJ_ARITH_PK_NEG_LO_B << (element * 3))) ^ (Op == PackedU64::Sub);
    const bool difference = negative_a != negative_b;
    const bool clamp = mode & RJ_ARITH_PK_CLAMP;
    const int reg = 2 * element;
#if defined(__AVX2__)
    for (int lane = 0; lane < 32; lane += kPackedLanes) {
      auto x = packed_pair(a + reg, lane);
      PackedWord value;
      if constexpr (Op == PackedU64::ShiftAdd) {
        auto shifts = packed_words(b[element] + lane), addend = packed_pair(c + reg, lane);
#if defined(__AVX512F__)
        value = _mm512_add_epi64(_mm512_sllv_epi64(x, shifts), addend);
#else
        value = _mm256_add_epi64(_mm256_sllv_epi64(x, shifts), addend);
#endif
      } else {
        auto y = packed_pair(b + reg, lane);
        // Opposite signs reduce to unsigned subtraction, possibly with operands swapped.
        value = negative_a && difference ? packed_sum(y, x, true, false, clamp)
                                         : packed_sum(x, y, difference, negative_a, clamp);
      }
      packed_stage(staged[reg] + lane, staged[reg + 1] + lane, value);
    }
#else
    for (int lane = 0; lane < 32; ++lane) {
      const uint64_t x = uint64_t(a[reg][lane]) | (uint64_t(a[reg + 1][lane]) << 32);
      uint64_t value;
      if constexpr (Op == PackedU64::ShiftAdd) {
        // Only active counts are constrained by the API domain; inactive data is immaterial.
        const uint64_t z = uint64_t(c[reg][lane]) | (uint64_t(c[reg + 1][lane]) << 32);
        value = (x << (b[element][lane] & 63)) + z;
      } else {
        const uint64_t y = uint64_t(b[reg][lane]) | (uint64_t(b[reg + 1][lane]) << 32);
        const uint64_t left = negative_a && difference ? y : x;
        const uint64_t right = negative_a && difference ? x : y;
        value = difference ? left - right : left + right;
        if (clamp) {
          if (difference) {
            value = left < right ? 0 : value;
          } else if (negative_a) {
            value = 0;
          } else if (value < left) {
            value = UINT64_MAX;
          }
        } else if (negative_a && !difference) {
          value = 0 - value;
        }
      }
      staged[reg][lane] = uint32_t(value);
      staged[reg + 1][lane] = uint32_t(value >> 32);
    }
#endif
  }
  // Fixed work regardless of mask density; publish logical registers in increasing order.
  for (int reg = 0; reg < 4; ++reg) {
#if defined(__AVX512F__)
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(staged[reg] + lane));
    }
#elif defined(__AVX2__)
    for (int lane = 0; lane < 32; lane += 8) {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), active,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane)));
    }
#else
    for (int lane = 0; lane < 32; ++lane) {
      if (exec_mask >> lane & 1) {
        d[reg][lane] = staged[reg][lane];
      }
    }
#endif
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
