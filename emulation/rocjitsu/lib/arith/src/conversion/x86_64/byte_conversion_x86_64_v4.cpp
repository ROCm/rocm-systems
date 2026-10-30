// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 byte-to-FP32 conversion, signed nibble offsets, and FP32-to-byte packing.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/byte_conversion.h"
#include "conversion/float_conversion.h"
#include "conversion/x86_64/lanes.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Byte>
void byte_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(a + lane);
    raw = _mm512_and_si512(_mm512_srli_epi32(raw, 8 * Byte), _mm512_set1_epi32(255));
    auto value = _mm512_mul_ps(_mm512_cvtepi32_ps(raw), scale);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm512_min_ps(value, _mm512_set1_ps(1));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(value));
  }
}

template void byte_conversion_x86_64_v4<0>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<1>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<2>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<3>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void nibble_offset_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {0.0625f, 0.125f, 0.25f, 0.03125f};
  auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto nibble = _mm512_srai_epi32(_mm512_slli_epi32(raw, 28), 28);
    auto value = _mm512_mul_ps(_mm512_cvtepi32_ps(nibble), scale);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
    }
    auto result = _mm512_castps_si512(value);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void byte_pack_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  using V = conversion_simd::Lanes<32>;
  const auto operation = conversion_detail::byte_pack_policy(flags, mode);
  for (int lane = 0; lane < 32; lane += V::size()) {
    V result = operation(V(a + lane), V(b + lane), V(c + lane));
    result.store(d + lane, exec_mask >> lane);
  }
}

} // namespace rocjitsu::arith
