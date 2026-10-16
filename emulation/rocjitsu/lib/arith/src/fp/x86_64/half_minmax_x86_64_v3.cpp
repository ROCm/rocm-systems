// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 three-input minimum, maximum, median, and combined selections.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/half_x86_64_v3.h"
#include "fp/half_minmax.h"
#include "fp/x86_64/minmax_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select among three FP16 values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Minmax3 Op>
void half_minmax3_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  using Policy = rocjitsu::arith::MinmaxPolicy<Op>;
  constexpr bool FirstMaximum = Policy::first_maximum;
  constexpr bool SecondMaximum = Policy::second_maximum;
  constexpr bool Propagate = Policy::propagate;
  constexpr bool Rdna3 = Policy::rdna3;
  constexpr bool Median = Policy::median;
  constexpr bool preserve_nan_inputs = Rdna3 || Propagate;

  int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false, preserve_nan_inputs>(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), a_shift, mode);
    auto y = half_input<false, preserve_nan_inputs>(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), b_shift, mode >> 1);
    auto z = half_input<false, preserve_nan_inputs>(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), c_shift, mode >> 2);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      // Replace subnormal FP32 lanes with signed zero.
      auto flush = [](__m256 input) {
        auto bits = _mm256_castps_si256(input);
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000),
                                       _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)));
        return _mm256_castsi256_ps(
            _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)), tiny));
      };
      x = flush(x);
      y = flush(y);
      z = flush(z);
    }
    __m256 value;
    if constexpr (Rdna3) {
      value = minmax3_rdna3<FirstMaximum, SecondMaximum, Median>(x, y, z,
                                                                 flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value = minmax3_value<FirstMaximum, SecondMaximum, Propagate, Median, true>(x, y, z);
    }
    if (mode & RJ_ARITH_ALU_OMOD_HALF) {
      value = _mm256_mul_ps(prepare_omod_f16(value, mode), scale);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      value = Rdna3 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)
                  ? _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q))
                  : clamped;
    }
    auto result = _mm256_sll_epi32(half_narrow<false, Rdna3>(value, flags & RJ_ARITH_FP16_OVFL),
                                   _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void half_minmax3_x86_64_v3<Minmax3::Min3Num>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::Max3Num>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *,
                                                       const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MinmaxNum>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *,
                                                         const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MaxminNum>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *,
                                                         const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::Minimum3>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *,
                                                        const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::Maximum3>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *,
                                                        const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MinimumMaximum>(uint64_t, uint32_t, uint32_t,
                                                              uint32_t *, const uint32_t *,
                                                              const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MaximumMinimum>(uint64_t, uint32_t, uint32_t,
                                                              uint32_t *, const uint32_t *,
                                                              const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MedianNum>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *,
                                                         const uint32_t *);

template void half_minmax3_x86_64_v3<Minmax3::Min3Legacy>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *,
                                                          const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::Max3Legacy>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *,
                                                          const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MinmaxLegacy>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *,
                                                            const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MaxminLegacy>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *,
                                                            const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<Minmax3::MedianLegacy>(uint64_t, uint32_t, uint32_t,
                                                            uint32_t *, const uint32_t *,
                                                            const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
