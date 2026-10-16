// SPDX-License-Identifier: MIT

// AVX2 FP16 exponent scaling and exponent/mantissa decomposition.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/half_x86_64_v3.h"
#include "fp/half_exponent.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Scale or decompose FP16 exponents for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <bool Ldexp, bool GuestDenormals>
void run(uint64_t flags, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
         const uint32_t *a, const uint32_t *b) {
  int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i result;
    if constexpr (Ldexp) {
      auto x = half_input<false>(raw, a_shift, mode);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        // Half subnormals widen to normal FP32 numbers; classify at the FP16 boundary.
        auto bits = _mm256_castps_si256(x);
        auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude);
        auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN));
        x = _mm256_castsi256_ps(_mm256_blendv_epi8(bits, sign, tiny));
      }
      auto exponent = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      exponent = _mm256_sra_epi32(_mm256_sll_epi32(exponent, _mm_cvtsi32_si128(16 - b_shift)),
                                  _mm_cvtsi32_si128(16));
      exponent = _mm256_min_epi32(_mm256_max_epi32(exponent, _mm256_set1_epi32(-64)),
                                  _mm256_set1_epi32(64));

      // Every nonzero half widened to FP32, and every product with this power
      // of two, is normal and exact. The architectural FP16 rounding precedes OMOD.
      auto factor = _mm256_castsi256_ps(
          _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23));
      auto value = _mm256_mul_ps(x, factor);
      if (mode & RJ_ARITH_ALU_OMOD_HALF) {
        value = prepare_omod_f16(value, mode);
        value = half_input<false>(half_narrow<false>(value, saturate), 0, 0);
        value = _mm256_mul_ps(value, scale);
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
        value = GuestDenormals && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)
                    ? _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q))
                    : clamped;
      }
      result = half_narrow<false>(value, saturate);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        auto magnitude = _mm256_and_si256(result, _mm256_set1_epi32(0x7fff));
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude);
        auto sign = _mm256_and_si256(result, _mm256_set1_epi32(0x8000));
        result = _mm256_blendv_epi8(result, sign, tiny);
      }
    } else {
      auto magnitude = _mm256_and_si256(_mm256_srl_epi32(raw, _mm_cvtsi32_si128(a_shift)),
                                        _mm256_set1_epi32(0x7fff));
      auto field = _mm256_srli_epi32(magnitude, 10);
      auto fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32(1023));

      // Converting the integer fraction to FP32 is exact in every rounding
      // mode and raises no FP exceptions. Its exponent identifies the top bit.
      auto top_bit = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(fraction)), 23);
      auto subnormal_exponent = _mm256_sub_epi32(top_bit, _mm256_set1_epi32(150));
      result =
          _mm256_blendv_epi8(_mm256_sub_epi32(field, _mm256_set1_epi32(14)), subnormal_exponent,
                             _mm256_cmpeq_epi32(field, _mm256_setzero_si256()));
      auto special = _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                                     _mm256_cmpeq_epi32(field, _mm256_set1_epi32(31)));
      result = _mm256_and_si256(_mm256_andnot_si256(special, result), _mm256_set1_epi32(0xffff));
    }
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Scale or decompose FP16 exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Ldexp>
void half_exponent_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                             const uint32_t *a, const uint32_t *b) {
  run<Ldexp, false>(0, saturate, exec_mask, mode, d, a, b);
}

// Scale or decompose FP16 exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_ldexp_guest_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                const uint32_t *a, const uint32_t *b) {
  run<true, true>(flags, flags & RJ_ARITH_FP16_OVFL, exec_mask, mode, d, a, b);
}

template void half_exponent_x86_64_v3<true>(bool, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                            const uint32_t *);
template void half_exponent_x86_64_v3<false>(bool, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *);

} // namespace rocjitsu::arith
