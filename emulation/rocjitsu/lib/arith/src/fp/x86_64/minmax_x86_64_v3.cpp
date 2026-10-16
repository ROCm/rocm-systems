// SPDX-License-Identifier: MIT

// AVX2 vector floating-point selection with GPU NaN and signed-zero policies.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/x86_64/minmax_x86_64_v3.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/minmax.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Select FP32 minima, maxima, or medians for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool Rdna3 = false, bool FlushInput = false>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  // Dispatch the guest input-denormal policy once, outside the four SIMD chunks.
  if constexpr (!FlushInput) {
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      return run<FirstMaximum, SecondMaximum, Propagate, Median, Rdna3, true>(flags, exec_mask,
                                                                              mode, d, a, b, c);
    }
  }
  const auto ka = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  const auto kb = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  const auto kc = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_C ? INT32_MAX : -1);
  const auto na = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  const auto nb = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  const auto nc = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), ka), na));
    auto y = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), kb), nb));
    auto z = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), kc), nc));
    if constexpr (FlushInput) {
      x = flush_denorm_f32(x);
      y = flush_denorm_f32(y);
      z = flush_denorm_f32(z);
    }
    __m256 value;
    if constexpr (Rdna3) {
      value = minmax3_rdna3<FirstMaximum, SecondMaximum, Median>(x, y, z,
                                                                 flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value = minmax3_value<FirstMaximum, SecondMaximum, Propagate, Median>(x, y, z);
    }
    if (mode & RJ_ARITH_ALU_OMOD_HALF) {
      value = prepare_omod_f32(value, mode);
      value = _mm256_mul_ps(value, scale);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      value = Rdna3 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)
                  ? _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q))
                  : clamped;
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace

// Select FP32 minima, maxima, or medians. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void minmax3_x86_64_v3(Minmax3 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  switch (op) {
  case Minmax3::Min3Legacy:
    return run<false, false, false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Legacy:
    return run<true, true, false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxLegacy:
    return run<false, true, false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminLegacy:
    return run<true, false, false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianLegacy:
    return run<false, false, false, true, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianNum:
    return run<false, false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Min3Num:
    return run<false, false, false>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Num:
    return run<true, true, false>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxNum:
    return run<false, true, false>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminNum:
    return run<true, false, false>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Minimum3:
    return run<false, false, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Maximum3:
    return run<true, true, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinimumMaximum:
    return run<false, true, true>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaximumMinimum:
    return run<true, false, true>(flags, exec_mask, mode, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
