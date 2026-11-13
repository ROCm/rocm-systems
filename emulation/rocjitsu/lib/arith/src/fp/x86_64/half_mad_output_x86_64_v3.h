// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared vector narrowing and OMOD/CLAMP for FP32-intermediate half MAD.
// AVX2 and AVX512 callers share this eight-lane output stage.

#pragma once

#include "common/x86_64/half_x86_64_v3.h"
#include "fp/x86_64/minmax_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Round the FP32 result to half, then apply inherited output scaling and clamp.
inline __m256i finish_half_mad(__m256 value, uint32_t mode, uint64_t flags) {
  auto half = half_narrow(value, flags & RJ_ARITH_FP16_OVFL);
  if (!(mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP))) {
    return half;
  }
  value = half_input(half, 0, 0);
  if (mode & RJ_ARITH_ALU_OMOD_MASK) {
    const float scales[] = {1, 2, 4, 0.5f};
    value = prepare_omod_f16(value, mode);
    auto scaled = _mm256_mul_ps(value, _mm256_set1_ps(scales[(mode >> 6) & 3]));
    value = selection_omod_round<16>(value, scaled, flags);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      clamped = _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q));
    }
    value = clamped;
  }
  return half_narrow(value, flags & RJ_ARITH_FP16_OVFL);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
