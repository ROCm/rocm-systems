// SPDX-License-Identifier: MIT

// AVX-512 FP16/FP32 conversion to signed/unsigned normalized 16-bit encodings.
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
__m512i normalized(__m512i raw, uint32_t mode, bool flush_input) {
  if constexpr (Half) {
    raw = _mm512_and_si512(
        _mm512_srl_epi32(raw, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
        _mm512_set1_epi32(65535));
    if (flush_input) {
      auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7fff));
      auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
      raw = _mm512_mask_mov_epi32(raw, tiny, _mm512_and_si512(raw, _mm512_set1_epi32(0x8000)));
    }
    raw = _mm512_castps_si512(_mm512_cvtph_ps(_mm512_cvtepi32_epi16(raw)));
  }
  raw = _mm512_and_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
  raw = _mm512_xor_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
  auto value = _mm512_castsi512_ps(raw);
  auto nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
  value = _mm512_min_ps(_mm512_max_ps(value, _mm512_set1_ps(Unsigned ? 0 : -1)), _mm512_set1_ps(1));
  value = _mm512_mask_mov_ps(value, nan, _mm512_setzero_ps());
  auto scale = _mm512_set1_ps(Unsigned ? 65535 : 32767);
  auto product = _mm512_mul_ps(value, scale);

  // Borrow rocjitsu's FMA residual correction for FP32 products that round
  // onto an integer midpoint. The final conversion then rounds only once.
  auto residual = _mm512_fmsub_ps(value, scale, product);
  auto lower = _mm512_roundscale_ps(product, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  auto rounded = _mm512_roundscale_ps(product, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  auto midpoint =
      _mm512_cmp_ps_mask(product, _mm512_add_ps(lower, _mm512_set1_ps(0.5f)), _CMP_EQ_OQ);
  auto down = midpoint & _mm512_cmp_ps_mask(residual, _mm512_setzero_ps(), _CMP_LT_OQ);
  auto up = midpoint & _mm512_cmp_ps_mask(residual, _mm512_setzero_ps(), _CMP_GT_OQ);
  rounded = _mm512_mask_mov_ps(rounded, down, lower);
  rounded = _mm512_mask_add_ps(rounded, up, lower, _mm512_set1_ps(1));
  return _mm512_and_si512(_mm512_cvttps_epi32(rounded), _mm512_set1_epi32(65535));
}

} // namespace

// Convert to normalized 16-bit values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Unsigned, NormalizedForm Form>
void normalized_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, bool flush_input) {
  constexpr bool unary = Form == NormalizedForm::Half;
  constexpr bool half = Form != NormalizedForm::PackedFloat;
  auto shift = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto result = normalized<Unsigned, half>(raw, mode, flush_input);
    if constexpr (unary) {
      auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      auto selected = _mm512_sll_epi32(_mm512_set1_epi32(65535), shift);
      result =
          _mm512_or_si512(_mm512_andnot_si512(selected, original), _mm512_sll_epi32(result, shift));
    } else {
      auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      result = _mm512_or_si512(
          result, _mm512_slli_epi32(normalized<Unsigned, half>(vb, mode >> 1, flush_input), 16));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void normalized_x86_64_v4<false, NormalizedForm::PackedFloat>(uint32_t, uint32_t,
                                                                       uint32_t *, const uint32_t *,
                                                                       const uint32_t *, bool);
template void normalized_x86_64_v4<true, NormalizedForm::PackedFloat>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *, bool);
template void normalized_x86_64_v4<false, NormalizedForm::PackedHalf>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *, bool);
template void normalized_x86_64_v4<true, NormalizedForm::PackedHalf>(uint32_t, uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *, bool);
template void normalized_x86_64_v4<false, NormalizedForm::Half>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *,
                                                                bool);
template void normalized_x86_64_v4<true, NormalizedForm::Half>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               bool);

} // namespace rocjitsu::arith
