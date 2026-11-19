// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP64 integral rounding with guest output controls and masked stores.
// Each lane's low/high VGPR words are captured before any result is published.

#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/fp64.h"
#include "fp/x86_64/fp64_output_x86_64_v4.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Round eight split-word lanes at a time, preserving aliases and inactive lanes.
template <int Rounding>
void round(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
           const uint32_t *const *a, bool quiet_snan) {
  const auto keep = _mm512_set1_epi64(mode & RJ_ARITH_ALU_ABS_A ? INT64_MAX : -1);
  const auto flip = _mm512_set1_epi64(mode & RJ_ARITH_ALU_NEG_A ? INT64_MIN : 0);
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; lane += 8) {
    auto low =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
    auto high =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)));
    auto bits = _mm512_xor_si512(
        _mm512_and_si512(_mm512_or_si512(low, _mm512_slli_epi64(high, 32)), keep), flip);
    auto x = _mm512_castsi512_pd(bits);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = flush_denorm_f64(x);
    }
    // Immediate rounding controls apply only to this instruction, never to MXCSR.
    auto value = _mm512_roundscale_pd(x, Rounding | _MM_FROUND_NO_EXC);
    const auto nan = _mm512_cmpgt_epi64_mask(_mm512_and_si512(bits, _mm512_set1_epi64(INT64_MAX)),
                                             _mm512_set1_epi64(0x7ff0000000000000LL));
    value =
        _mm512_mask_mov_pd(value, nan,
                           _mm512_castsi512_pd(_mm512_or_si512(
                               bits, _mm512_set1_epi64(quiet_snan ? 0x0008000000000000LL : 0))));
    bits = _mm512_castpd_si512(fp64_output_x86_64_v4(
        value, mode, flags, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), _mm512_cvtepi64_epi32(bits));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
  }
  // A destination word may overlap either source word; publish only after all reads.
  for (int word = 0; word < 2; ++word) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[word] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[word] + lane));
    }
  }
}

} // namespace

// Apply the selected integral rounding rule and validated guest controls.
void round_f64_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *const *d, const uint32_t *const *a, bool quiet_snan) {
  switch (op) {
  case Fp64::Trunc:
    return round<_MM_FROUND_TO_ZERO>(flags, exec_mask, mode, d, a, quiet_snan);
  case Fp64::Ceil:
    return round<_MM_FROUND_TO_POS_INF>(flags, exec_mask, mode, d, a, quiet_snan);
  case Fp64::Rndne:
    return round<_MM_FROUND_TO_NEAREST_INT>(flags, exec_mask, mode, d, a, quiet_snan);
  case Fp64::Floor:
    return round<_MM_FROUND_TO_NEG_INF>(flags, exec_mask, mode, d, a, quiet_snan);
  default:
    __builtin_unreachable();
  }
}

} // namespace rocjitsu::arith
