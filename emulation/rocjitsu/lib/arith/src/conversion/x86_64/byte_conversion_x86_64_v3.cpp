// SPDX-License-Identifier: MIT

// AVX2 byte-to-FP32 conversion, signed nibble offsets, and FP32-to-byte packing.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/byte_conversion.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Byte>
void byte_conversion_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    raw = _mm256_and_si256(_mm256_srli_epi32(raw, 8 * Byte), _mm256_set1_epi32(255));
    auto value = _mm256_mul_ps(_mm256_cvtepi32_ps(raw), scale);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm256_min_ps(value, _mm256_set1_ps(1));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

template void byte_conversion_x86_64_v3<0>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<1>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<2>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<3>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void nibble_offset_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {0.0625f, 0.125f, 0.25f, 0.03125f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto nibble = _mm256_srai_epi32(_mm256_slli_epi32(raw, 28), 28);
    auto value = _mm256_mul_ps(_mm256_cvtepi32_ps(nibble), scale);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    }
    auto result = _mm256_castps_si256(value);
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void byte_pack_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b, const uint32_t *c) {
  auto keep = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto flip = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto value = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_and_si256(va, keep), flip));
    value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(255));
    value = _mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto byte = _mm256_cvttps_epi32(value);
    auto shift = _mm256_slli_epi32(_mm256_and_si256(vb, _mm256_set1_epi32(3)), 3);
    auto selected = _mm256_sllv_epi32(_mm256_set1_epi32(255), shift);
    auto result =
        _mm256_or_si256(_mm256_andnot_si256(selected, vc), _mm256_sllv_epi32(byte, shift));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace rocjitsu::arith
