// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 pairwise FP32 packing into FP16/BF16 with format-specific rounding.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/float_conversion.h"
#include "conversion/packed_conversion.h"
#include "rj_arith/rj_arith.h"
#include <experimental/simd>

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply source modifiers and convert FP32 to the selected packed result format.
template <PackedConversion Op> __m256i narrow(__m256i raw, uint32_t mode) {
  raw = _mm256_and_si256(raw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
  raw = _mm256_xor_si256(raw, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
  auto value = _mm256_castsi256_ps(raw);
  if constexpr (Op == PackedConversion::HalfRtz) {
    return _mm256_cvtepu16_epi32(_mm256_cvtps_ph(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  } else if constexpr (Op == PackedConversion::Unsigned) {
    value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(65535));
    return _mm256_cvttps_epi32(value);
  } else {
    auto nan = _mm256_castps_si256(_mm256_cmp_ps(value, value, _CMP_UNORD_Q));
    value = _mm256_min_ps(_mm256_max_ps(value, _mm256_set1_ps(-32768)), _mm256_set1_ps(32767));
    auto result = _mm256_and_si256(_mm256_cvttps_epi32(value), _mm256_set1_epi32(65535));
    return _mm256_andnot_si256(nan, result);
  }
}

} // namespace

// Pack two FP32 conversions into one word. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <PackedConversion Op>
void packed_conversion_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                 const uint32_t *a, const uint32_t *b) {
  if constexpr (Op == PackedConversion::HalfRtz) {
    namespace sx = std::experimental;
    using V = sx::native_simd<uint32_t>;
    const auto low =
        conversion_detail::float_policy<conversion_detail::fmt::F32, conversion_detail::fmt::F16>(
            (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO,
            mode & (RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A));
    const auto high =
        conversion_detail::float_policy<conversion_detail::fmt::F32, conversion_detail::fmt::F16>(
            (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO,
            (mode >> 1) & (RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A));
    for (int lane = 0; lane < 32; lane += V::size()) {
      V result =
          low(V(a + lane, sx::element_aligned)) | (high(V(b + lane, sx::element_aligned)) << 16);
      V indices([](auto i) { return uint32_t(i); });
      auto active = ((V(exec_mask >> lane) >> indices) & V(1U)) != V(0U);
      sx::where(active, result).copy_to(d + lane, sx::element_aligned);
    }
    return;
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto result =
        _mm256_or_si256(narrow<Op>(va, mode), _mm256_slli_epi32(narrow<Op>(vb, mode >> 1), 16));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void packed_conversion_x86_64_v3<PackedConversion::HalfRtz>(uint64_t, uint32_t, uint32_t,
                                                                     uint32_t *, const uint32_t *,
                                                                     const uint32_t *);
template void packed_conversion_x86_64_v3<PackedConversion::Signed>(uint64_t, uint32_t, uint32_t,
                                                                    uint32_t *, const uint32_t *,
                                                                    const uint32_t *);
template void packed_conversion_x86_64_v3<PackedConversion::Unsigned>(uint64_t, uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);

} // namespace rocjitsu::arith
