// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX512 mixed BF16 FMA: sixteen FP32 lanes or eight compensated FP64 sums.
#include "fp/mixed_bf16.h"
#include "fp/x86_64/mixed_bf16_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

// Execute validated mixed BF16 controls without whole-vector exceptional fallbacks.
void rocjitsu::arith::mixed_bf16_x86_64_v4(MixedFma destination, uint64_t flags, uint32_t exec_mask,
                                           uint32_t mode, uint32_t *d, const uint32_t *a,
                                           const uint32_t *b, const uint32_t *c) {
  const bool clamp = mode & RJ_ARITH_ALU_CLAMP;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const bool loose = (flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if (loose || destination == MixedFma::Float) {
    auto input = [](const uint32_t *source, uint32_t mode) {
      auto bits = _mm512_loadu_si512(source);
      if (mode & RJ_ARITH_MIX_BF16_A) {
        bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm512_and_si512(bits, _mm512_set1_epi32(-65536))
                                          : _mm512_slli_epi32(bits, 16);
      }
      if (mode & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      }
      if (mode & RJ_ARITH_ALU_NEG_A) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi32(INT32_MIN));
      }
      return _mm512_castsi512_ps(bits);
    };
    for (int lane = 0; lane < 32; lane += 16) {
      auto x = input(a + lane, mode), y = input(b + lane, mode >> 1),
           z = input(c + lane, mode >> 2);
      auto result = _mm512_fmadd_ps(x, y, z);
      if (clamp) {
        result = _mm512_min_ps(_mm512_max_ps(result, _mm512_setzero_ps()), _mm512_set1_ps(1));
      }
      unsigned special = !loose
                             ? (_mm512_fpclass_ps_mask(x, 0x99) | _mm512_fpclass_ps_mask(y, 0x99) |
                                _mm512_fpclass_ps_mask(z, 0x99)) &
                                   (exec_mask >> lane)
                             : 0;
      if (special) {
        // A lone NaN with two finite operands has unambiguous hardware FMA payload.
        auto nx = _mm512_fpclass_ps_mask(x, 0x81), ny = _mm512_fpclass_ps_mask(y, 0x81),
             nz = _mm512_fpclass_ps_mask(z, 0x81);
        auto inf = _mm512_fpclass_ps_mask(x, 0x18) | _mm512_fpclass_ps_mask(y, 0x18) |
                   _mm512_fpclass_ps_mask(z, 0x18);
        special &= inf | (nx & ny) | (ny & nz) | (nx & nz);
      }
      if (special) {
        alignas(64) uint32_t av[16], bv[16], cv[16], out[16];
        _mm512_store_ps(av, x);
        _mm512_store_ps(bv, y);
        _mm512_store_ps(cv, z);
        _mm512_store_ps(out, result);
        for (; special; special &= special - 1) {
          int i = __builtin_ctz(special);
          out[i] = mixed_bf16_value(destination, flags, clamp, av[i], bv[i], cv[i]);
        }
        result = _mm512_load_ps(out);
      }
      auto words = _mm512_castps_si512(result);
      if (destination != MixedFma::Float) {
        // Integer nearest-even narrowing avoids a BF16-only CPU specialization.
        auto increment =
            _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                             _mm512_and_si512(_mm512_srli_epi32(words, 16), _mm512_set1_epi32(1)));
        auto rounded = _mm512_srli_epi32(_mm512_add_epi32(words, increment), 16);
        const auto nan = _mm512_cmpgt_epi32_mask(
            _mm512_and_si512(words, _mm512_set1_epi32(INT32_MAX)), _mm512_set1_epi32(0x7f800000));
        words = _mm512_mask_mov_epi32(
            rounded, nan, _mm512_or_si512(_mm512_srli_epi32(words, 16), _mm512_set1_epi32(0x40)));
        const bool high = destination == MixedFma::High;
        if (high) {
          words = _mm512_slli_epi32(words, 16);
        }
        words = _mm512_or_si512(words, _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                                        _mm512_set1_epi32(high ? 0xffff : -65536)));
      }
      _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), words);
    }
    return;
  }
  for (int lane = 0; lane < 32; lane += 8) {
    const auto x = mixed_bf16_input8(a + lane, mode);
    const auto y = mixed_bf16_input8(b + lane, mode >> 1);
    const auto z = mixed_bf16_input8(c + lane, mode >> 2);
    const auto product = _mm512_mul_pd(_mm512_cvtps_pd(x), _mm512_cvtps_pd(y));
    const auto addend = _mm512_cvtps_pd(z);
    const auto sum = _mm512_add_pd(product, addend);
    const auto virtual_z = _mm512_sub_pd(sum, product);
    const auto error = _mm512_add_pd(_mm512_sub_pd(product, _mm512_sub_pd(sum, virtual_z)),
                                     _mm512_sub_pd(addend, virtual_z));
    const auto lo = mixed_bf16_narrow4(_mm512_castpd512_pd256(sum), _mm512_castpd512_pd256(error),
                                       _mm512_castpd512_pd256(product),
                                       _mm512_castpd512_pd256(addend), rounding, clamp);
    const auto hi = mixed_bf16_narrow4(
        _mm512_extractf64x4_pd(sum, 1), _mm512_extractf64x4_pd(error, 1),
        _mm512_extractf64x4_pd(product, 1), _mm512_extractf64x4_pd(addend, 1), rounding, clamp);
    auto result = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    result = mixed_bf16_finish8(destination, flags, clamp, x, y, z, result);
    result = mixed_bf16_merge8(destination, result, d + lane);
    _mm256_mask_storeu_epi32(d + lane, __mmask8(exec_mask >> lane), result);
  }
}
