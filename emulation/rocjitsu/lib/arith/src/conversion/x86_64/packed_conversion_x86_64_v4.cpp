// SPDX-License-Identifier: MIT

// AVX-512 pairwise FP32 packing into FP16/BF16 with format-specific rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/packed_conversion.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply source modifiers and convert FP32 to the selected packed result format.
template <PackedConversion Op> __m512i narrow(__m512i raw, uint32_t mode) {
  raw = _mm512_and_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
  raw = _mm512_xor_si512(raw, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
  auto value = _mm512_castsi512_ps(raw);
  if constexpr (Op == PackedConversion::HalfRtz) {
    return _mm512_cvtepu16_epi32(_mm512_cvtps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  } else if constexpr (Op == PackedConversion::Unsigned) {
    value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(65535));
    return _mm512_cvttps_epi32(value);
  } else {
    auto nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
    value = _mm512_min_ps(_mm512_max_ps(value, _mm512_set1_ps(-32768)), _mm512_set1_ps(32767));
    auto result = _mm512_and_si512(_mm512_cvttps_epi32(value), _mm512_set1_epi32(65535));
    return _mm512_mask_mov_epi32(result, nan, _mm512_setzero_si512());
  }
}

} // namespace

// Pack two FP32 conversions into one word. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <PackedConversion Op>
void packed_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    auto result =
        _mm512_or_si512(narrow<Op>(va, mode), _mm512_slli_epi32(narrow<Op>(vb, mode >> 1), 16));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void packed_conversion_x86_64_v4<PackedConversion::HalfRtz>(uint32_t, uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void packed_conversion_x86_64_v4<PackedConversion::Signed>(uint32_t, uint32_t, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void packed_conversion_x86_64_v4<PackedConversion::Unsigned>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);

} // namespace rocjitsu::arith
