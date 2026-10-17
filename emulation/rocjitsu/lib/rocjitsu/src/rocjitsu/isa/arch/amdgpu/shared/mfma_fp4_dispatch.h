// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__FMA__)
#include <xmmintrin.h>
#endif

namespace rocjitsu::amdgpu {

// Keep the specialized shape contract shared by execution and admission.
constexpr bool mfma_fp4_avx512_shape(uint32_t m, uint32_t n, uint32_t k, uint32_t batches,
                                     uint32_t a_bits, uint32_t b_bits, uint32_t wave_size) {
  return m == 16 && n == 16 && k == 128 && batches == 1 && a_bits == 4 && b_bits == 4 &&
         wave_size == 64;
}

// Integer dot products reconstruct zero as +0. Host RNE gives the same sign
// as finite FP4 accumulation from +0. This query contains no kernel bodies.
inline bool mfma_fp4_vnni_available() {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__FMA__)
  return __builtin_cpu_supports("avx512vnni") &&
         (_mm_getcsr() & _MM_ROUND_MASK) == _MM_ROUND_NEAREST;
#else
  return false;
#endif
}

inline bool mfma_fp4_vnni_selected(uint32_t m, uint32_t n, uint32_t k, uint32_t batches,
                                   uint32_t a_bits, uint32_t b_bits, uint32_t wave_size,
                                   bool force_scalar) {
  return !force_scalar && mfma_fp4_avx512_shape(m, n, k, batches, a_bits, b_bits, wave_size) &&
         mfma_fp4_vnni_available();
}

} // namespace rocjitsu::amdgpu
