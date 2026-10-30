// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose reciprocal with fixed denormal flushing and output modifiers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "math/rcp_iflag.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Loose reciprocal with fixed input/output denormal flushing and no FP status.
// Preserve inactive lanes and support D/A aliasing.
// Requires the named CPU features and validated instruction controls.
void rcp_iflag_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a) {
  auto exponent = _mm256_set1_epi32(0x7f800000), sign = _mm256_set1_epi32(int(0x80000000u)),
       zero = _mm256_setzero_si256();
  auto keep = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : -1),
       flip = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? int(0x80000000u) : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto tiny = _mm256_cmpeq_epi32(_mm256_and_si256(raw, exponent), zero);
    raw = _mm256_xor_si256(_mm256_and_si256(raw, keep), flip);
    raw = _mm256_blendv_epi8(raw, _mm256_and_si256(raw, sign), tiny);
    auto value = _mm256_div_ps(_mm256_set1_ps(1), _mm256_castsi256_ps(raw));
    auto bits = _mm256_castps_si256(value);
    bits = _mm256_blendv_epi8(bits, _mm256_and_si256(bits, sign),
                              _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), zero));
    if (mode & RJ_ARITH_ALU_OMOD_MASK) {
      bits = _mm256_blendv_epi8(
          bits, zero,
          _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), zero));
      value = _mm256_mul_ps(_mm256_castsi256_ps(bits), scale);
    } else {
      value = _mm256_castsi256_ps(bits);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    bits = _mm256_castps_si256(value);
    bits = _mm256_blendv_epi8(bits, _mm256_and_si256(bits, sign),
                              _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), zero));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, bits);
  }
}

} // namespace rocjitsu::arith
