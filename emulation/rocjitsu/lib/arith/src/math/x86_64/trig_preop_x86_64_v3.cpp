// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 division of 2/pi into exponent-selected pieces for trigonometric range reduction.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/trig_preop.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select an exponent-dependent piece of 2/pi. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void trig_preop_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b) {
  constexpr unsigned valid_bits = Rdna3 ? 1201 : 1184;
  const uint64_t *table = Rdna3 ? trig_preop_table.data() : trig_preop_table_rdna4.data();
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; lane += 4) {
    auto a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a_hi + lane));
    auto selector = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b + lane)),
                                  _mm_set1_epi32(31));
    auto exponent = _mm_and_si128(_mm_srli_epi32(a, 20), _mm_set1_epi32(2047));
    auto shift = _mm_add_epi32(
        _mm_mullo_epi32(selector, _mm_set1_epi32(53)),
        _mm_max_epi32(_mm_sub_epi32(exponent, _mm_set1_epi32(1077)), _mm_setzero_si128()));
    auto bank = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1967)),
                              _mm_set1_epi32(valid_bits + 1));
    auto index = _mm_add_epi32(bank, _mm_min_epi32(shift, _mm_set1_epi32(valid_bits)));
    auto value = _mm256_i32gather_epi64(reinterpret_cast<const long long *>(table), index, 8);
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      if constexpr (Rdna3) {
        const auto hidden = _mm256_set1_epi64x(1ULL << 52);
        const auto tiny = _mm256_cmpgt_epi64(hidden, value);
        if (omod == 3) {
          auto rounded = _mm256_add_epi64(
              _mm256_srli_epi64(value, 1),
              _mm256_and_si256(_mm256_cmpeq_epi64(_mm256_and_si256(value, _mm256_set1_epi64x(3)),
                                                  _mm256_set1_epi64x(3)),
                               _mm256_set1_epi64x(1)));
          auto boundary = _mm256_cmpgt_epi64(_mm256_set1_epi64x(2ULL << 52), value);
          value = _mm256_blendv_epi8(_mm256_sub_epi64(value, hidden), rounded, boundary);
        } else {
          auto shifted = omod == 1 ? _mm256_slli_epi64(value, 1) : _mm256_slli_epi64(value, 2);
          if (omod == 2) {
            auto carry = _mm256_cmpgt_epi64(shifted, _mm256_set1_epi64x((2ULL << 52) - 1));
            shifted = _mm256_blendv_epi8(
                shifted, _mm256_add_epi64(_mm256_srli_epi64(shifted, 1), hidden), carry);
          }
          value = _mm256_blendv_epi8(
              _mm256_add_epi64(value, _mm256_set1_epi64x(uint64_t(omod) << 52)), shifted, tiny);
        }
        value =
            _mm256_and_si256(value, _mm256_cmpgt_epi64(value, _mm256_set1_epi64x(0xfffffffffffff)));
      } else {
        auto normal = _mm256_cmpgt_epi64(value, _mm256_set1_epi64x(0xfffffffffffff));
        value = _mm256_add_epi64(
            value, _mm256_set1_epi64x(int64_t(omod == 3 ? -1 : int(omod)) * (INT64_C(1) << 52)));
        normal = _mm256_and_si256(normal,
                                  _mm256_cmpgt_epi64(value, _mm256_set1_epi64x(0xfffffffffffff)));
        value = _mm256_and_si256(value, normal);
      }
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto one = _mm256_set1_epi64x(0x3ff0000000000000);
      value = _mm256_blendv_epi8(value, one, _mm256_cmpgt_epi64(value, one));
    }
    auto lo = _mm256_shuffle_epi32(value, _MM_SHUFFLE(2, 0, 2, 0));
    auto hi = _mm256_shuffle_epi32(value, _MM_SHUFFLE(3, 1, 3, 1));
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(result[0] + lane),
        _mm_unpacklo_epi64(_mm256_castsi256_si128(lo), _mm256_extracti128_si256(lo, 1)));
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(result[1] + lane),
        _mm_unpacklo_epi64(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi, 1)));
  }
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask, value);
    }
  }
}

template void trig_preop_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *,
                                          const uint32_t *);
template void trig_preop_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *,
                                         const uint32_t *);

} // namespace rocjitsu::arith
