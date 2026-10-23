// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 division of 2/pi into exponent-selected pieces for trigonometric range reduction.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/trig_preop.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select an exponent-dependent piece of 2/pi. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void trig_preop_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b) {
  constexpr unsigned valid_bits = Rdna3 ? 1201 : 1184;
  const uint64_t *table = Rdna3 ? trig_preop_table.data() : trig_preop_table_rdna4.data();
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; lane += 8) {
    auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_hi + lane));
    auto selector = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), _mm256_set1_epi32(31));
    auto exponent = _mm256_and_si256(_mm256_srli_epi32(a, 20), _mm256_set1_epi32(2047));
    auto shift =
        _mm256_add_epi32(_mm256_mullo_epi32(selector, _mm256_set1_epi32(53)),
                         _mm256_max_epi32(_mm256_sub_epi32(exponent, _mm256_set1_epi32(1077)),
                                          _mm256_setzero_si256()));
    auto bank = _mm256_and_si256(_mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(1967)),
                                 _mm256_set1_epi32(valid_bits + 1));
    auto index = _mm256_add_epi32(bank, _mm256_min_epi32(shift, _mm256_set1_epi32(valid_bits)));
    auto value = _mm512_i32gather_epi64(index, table, 8);
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      if constexpr (Rdna3) {
        const auto hidden = _mm512_set1_epi64(1ULL << 52);
        const auto tiny = _mm512_cmpgt_epi64_mask(hidden, value);
        if (omod == 3) {
          auto odd_half = _mm512_cmpeq_epi64_mask(_mm512_and_si512(value, _mm512_set1_epi64(3)),
                                                  _mm512_set1_epi64(3));
          auto rounded = _mm512_mask_add_epi64(_mm512_srli_epi64(value, 1), odd_half,
                                               _mm512_srli_epi64(value, 1), _mm512_set1_epi64(1));
          auto boundary = _mm512_cmpgt_epi64_mask(_mm512_set1_epi64(2ULL << 52), value);
          value = _mm512_mask_blend_epi64(boundary, _mm512_sub_epi64(value, hidden), rounded);
        } else {
          auto shifted = omod == 1 ? _mm512_slli_epi64(value, 1) : _mm512_slli_epi64(value, 2);
          if (omod == 2) {
            auto carry = _mm512_cmpgt_epi64_mask(shifted, _mm512_set1_epi64((2ULL << 52) - 1));
            shifted = _mm512_mask_blend_epi64(
                carry, shifted, _mm512_add_epi64(_mm512_srli_epi64(shifted, 1), hidden));
          }
          value = _mm512_mask_blend_epi64(
              tiny, _mm512_add_epi64(value, _mm512_set1_epi64(uint64_t(omod) << 52)), shifted);
        }
        value = _mm512_maskz_mov_epi64(
            _mm512_cmpgt_epi64_mask(value, _mm512_set1_epi64(0xfffffffffffff)), value);
      } else {
        auto normal = _mm512_cmpgt_epi64_mask(value, _mm512_set1_epi64(0xfffffffffffff));
        value = _mm512_add_epi64(
            value, _mm512_set1_epi64(int64_t(omod == 3 ? -1 : int(omod)) * (INT64_C(1) << 52)));
        normal &= _mm512_cmpgt_epi64_mask(value, _mm512_set1_epi64(0xfffffffffffff));
        value = _mm512_maskz_mov_epi64(normal, value);
      }
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm512_min_epi64(value, _mm512_set1_epi64(0x3ff0000000000000));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                        _mm512_cvtepi64_epi32(value));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(value, 32)));
  }
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

template void trig_preop_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *,
                                          const uint32_t *);
template void trig_preop_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *,
                                         const uint32_t *);

} // namespace rocjitsu::arith
