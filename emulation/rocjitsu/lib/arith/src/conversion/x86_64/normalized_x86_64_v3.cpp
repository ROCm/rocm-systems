// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16/FP32 conversion to signed/unsigned normalized 16-bit encodings.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/normalized.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Convert one selected FP16/FP32 value to normalized signed/unsigned 16-bit bits, honoring
// input flushing.
template <bool Unsigned, bool Half>
__m256i normalized(__m256i raw, uint32_t mode, bool flush_input) {
  if constexpr (Half) {
    raw = _mm256_and_si256(
        _mm256_srl_epi32(raw, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
        _mm256_set1_epi32(65535));
    if (flush_input) {
      auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fff));
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude);
      raw = _mm256_blendv_epi8(raw, _mm256_and_si256(raw, _mm256_set1_epi32(0x8000)), tiny);
    }
    auto halves = _mm_packus_epi32(_mm256_castsi256_si128(raw), _mm256_extracti128_si256(raw, 1));
    raw = _mm256_castps_si256(_mm256_cvtph_ps(halves));
  }
  raw = _mm256_and_si256(raw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
  raw = _mm256_xor_si256(raw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
  auto value = _mm256_castsi256_ps(raw);
  auto nan = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
  value = _mm256_min_ps(_mm256_max_ps(value, _mm256_set1_ps(Unsigned ? 0 : -1)), _mm256_set1_ps(1));
  value = _mm256_andnot_ps(nan, value);
  auto scale = _mm256_set1_ps(Unsigned ? 65535 : 32767);
  auto product = _mm256_mul_ps(value, scale);

  // Borrow rocjitsu's FMA residual correction for FP32 products that round
  // onto an integer midpoint. The final conversion then rounds only once.
  auto residual = _mm256_fmsub_ps(value, scale, product);
  auto lower = _mm256_round_ps(product, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  auto rounded = _mm256_round_ps(product, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  auto midpoint = _mm256_cmp_ps(product, _mm256_add_ps(lower, _mm256_set1_ps(0.5f)), _CMP_EQ_OQ);
  auto down = _mm256_and_ps(midpoint, _mm256_cmp_ps(residual, _mm256_setzero_ps(), _CMP_LT_OQ));
  auto up = _mm256_and_ps(midpoint, _mm256_cmp_ps(residual, _mm256_setzero_ps(), _CMP_GT_OQ));
  rounded = _mm256_blendv_ps(rounded, lower, down);
  rounded = _mm256_blendv_ps(rounded, _mm256_add_ps(lower, _mm256_set1_ps(1)), up);
  return _mm256_and_si256(_mm256_cvttps_epi32(rounded), _mm256_set1_epi32(65535));
}

} // namespace

// Convert to normalized 16-bit values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, NormalizedForm Form>
void normalized_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, bool flush_input) {
  constexpr bool unary = Form == NormalizedForm::Half;
  constexpr bool half = Form != NormalizedForm::PackedFloat;
  auto shift = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto result = normalized<Unsigned, half>(raw, mode, flush_input);
    if constexpr (unary) {
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      auto selected = _mm256_sll_epi32(_mm256_set1_epi32(65535), shift);
      result =
          _mm256_or_si256(_mm256_andnot_si256(selected, original), _mm256_sll_epi32(result, shift));
    } else {
      auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      result = _mm256_or_si256(
          result, _mm256_slli_epi32(normalized<Unsigned, half>(vb, mode >> 1, flush_input), 16));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void normalized_x86_64_v3<false, NormalizedForm::PackedFloat>(uint32_t, uint32_t,
                                                                       uint32_t *, const uint32_t *,
                                                                       const uint32_t *, bool);
template void normalized_x86_64_v3<true, NormalizedForm::PackedFloat>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *, bool);
template void normalized_x86_64_v3<false, NormalizedForm::PackedHalf>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *, bool);
template void normalized_x86_64_v3<true, NormalizedForm::PackedHalf>(uint32_t, uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *, bool);
template void normalized_x86_64_v3<false, NormalizedForm::Half>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *,
                                                                bool);
template void normalized_x86_64_v3<true, NormalizedForm::Half>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               bool);

} // namespace rocjitsu::arith
