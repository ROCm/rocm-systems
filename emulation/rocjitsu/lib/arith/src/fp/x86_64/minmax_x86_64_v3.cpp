// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 vector floating-point selection using the inherited NaN and signed-zero model.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/x86_64/minmax_x86_64_v3.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/minmax_impl.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Select FP32 minima, maxima, or medians for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <Minmax3 Op, bool FlushInput = false>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  using Policy = rocjitsu::arith::MinmaxPolicy<Op>;
  constexpr bool FirstMaximum = Policy::first_maximum;
  constexpr bool SecondMaximum = Policy::second_maximum;
  constexpr bool PropagateNaN = Policy::propagate;
  constexpr bool Rdna3 = Policy::rdna3;
  constexpr bool Median = Policy::median;

  // Dispatch the guest input-denormal policy once, outside the four SIMD chunks.
  if constexpr (!FlushInput) {
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      return run<Op, true>(flags, exec_mask, mode, d, a, b, c);
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
      value = minmax3_value<FirstMaximum, SecondMaximum, PropagateNaN, Median>(x, y, z);
    }
    if (mode & RJ_ARITH_ALU_OMOD_HALF) {
      value = prepare_omod_f32(value, mode);
      value = selection_omod_round<32>(value, _mm256_mul_ps(value, scale), flags);
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
    return run<Minmax3::Min3Legacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Legacy:
    return run<Minmax3::Max3Legacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxLegacy:
    return run<Minmax3::MinmaxLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminLegacy:
    return run<Minmax3::MaxminLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianLegacy:
    return run<Minmax3::MedianLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianNum:
    return run<Minmax3::MedianNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Min3Num:
    return run<Minmax3::Min3Num>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Num:
    return run<Minmax3::Max3Num>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxNum:
    return run<Minmax3::MinmaxNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminNum:
    return run<Minmax3::MaxminNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Minimum3:
    return run<Minmax3::Minimum3>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Maximum3:
    return run<Minmax3::Maximum3>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinimumMaximum:
    return run<Minmax3::MinimumMaximum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaximumMinimum:
    return run<Minmax3::MaximumMinimum>(flags, exec_mask, mode, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
