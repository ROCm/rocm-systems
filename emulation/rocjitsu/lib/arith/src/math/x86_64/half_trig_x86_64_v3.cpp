// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 sine/cosine with guest input/output modifiers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/validation.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "math/trig.h"
#include "math/x86_64/trig_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply half-precision output scaling and clamp to a rounded trig result.
__m256i output(__m256i bits, uint32_t mode, bool preserve_nan) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
  unsigned scale = (mode >> 6) & 3;
  if (scale) {
    __m256i scaled;
    if (scale == 3) {
      const __m256i rounded = _mm256_add_epi32(
          _mm256_srli_epi32(magnitude, 1),
          _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(magnitude, _mm256_set1_epi32(3)),
                                              _mm256_set1_epi32(3)),
                           _mm256_set1_epi32(1)));
      scaled = _mm256_blendv_epi8(_mm256_sub_epi32(magnitude, _mm256_set1_epi32(0x400)), rounded,
                                  _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800), magnitude));
    } else {
      scaled = _mm256_add_epi32(magnitude, _mm256_set1_epi32(int(scale * 0x400)));
    }
    // Already tiny operands become +0; newly tiny results retain their sign.
    scaled = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), scaled), scaled);
    scaled = _mm256_or_si256(scaled, _mm256_and_si256(bits, _mm256_set1_epi32(0x8000)));
    scaled = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude), scaled);
    bits =
        _mm256_blendv_epi8(bits, scaled, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7c00), magnitude));
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const __m256i negative = _mm256_cmpgt_epi32(bits, _mm256_set1_epi32(0x7fff));
    const __m256i nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7c00));
    auto clamped = _mm256_blendv_epi8(_mm256_min_epu32(bits, _mm256_set1_epi32(0x3c00)), zero,
                                      _mm256_or_si256(negative, nan));
    bits = preserve_nan ? _mm256_blendv_epi8(clamped, bits, nan) : clamped;
  }
  return bits;
}

// Evaluate FP16 sine/cosine for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool Cosine>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const __m256i keep = _mm256_set1_epi32(d_shift ? 0xffff : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto raw = _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(a_shift)),
                                _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    raw = _mm256_xor_si256(raw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
    auto input = half_input<false>(words, a_shift, mode);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400),
                                     _mm256_and_si256(raw, _mm256_set1_epi32(0x7fff)));
      input = _mm256_blendv_ps(input, _mm256_and_ps(input, _mm256_set1_ps(-0.0f)),
                               _mm256_castsi256_ps(tiny));
    }
    const __m256 value = trig_value<Cosine>(_mm256_castps_si256(input));
    auto result = half_narrow<false>(value, false);
    // One input magnitude per operation rounds one half ULP above the inherited
    // integer model. Repair it before modifiers, sharing arithmetic with Loose.
    auto repair =
        _mm256_and_si256(_mm256_set1_epi32(rocjitsu::arith::is_exact(flags) ? -1 : 0),
                         _mm256_cmpeq_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7fff)),
                                            _mm256_set1_epi32(Cosine ? 0x3251 : 0x2abc)));
    result = _mm256_add_epi32(result, repair);
    if (!(flags & RJ_ARITH_FP_IEEE_MODE)) {
      auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7fff)),
                                    _mm256_set1_epi32(0x7c00));
      result = _mm256_blendv_epi8(result, raw, nan);
    }
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400),
                                     _mm256_and_si256(result, _mm256_set1_epi32(0x7fff)));
      result =
          _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)), tiny);
    }
    result = output(result, mode, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    const __m256i previous = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(previous, keep));
    const __m256i lane_exec_mask =
        _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Evaluate FP16 sine/cosine. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_trig_x86_64_v3(bool cosine, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                         uint32_t *d, const uint32_t *a) {
  if (cosine) {
    run<true>(flags, exec_mask, mode, d, a);
  } else {
    run<false>(flags, exec_mask, mode, d, a);
  }
}

} // namespace rocjitsu::arith
