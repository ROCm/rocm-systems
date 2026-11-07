// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 BF16 math packing and modifiers around existing FP32 numeric backends.
#include "common/x86_64/alu_x86_64_v3.h"
#include "math/x86_64/bf16_math.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Stage the entire source before writing, preserving aliases, inactive lanes and other halves.
int bf16_math_x86_64_v3(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a) {
  uint32_t input[32], output[32]{};
  const auto keep_source = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : -1);
  const auto flip = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? int(0x80000000U) : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm256_and_si256(bits, _mm256_set1_epi32(-65536))
                                      : _mm256_slli_epi32(bits, 16);
    bits = _mm256_xor_si256(_mm256_and_si256(bits, keep_source), flip);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(input + lane), bits);
  }
  const uint32_t *sources[] = {input};
  uint32_t *destinations[] = {output};
  // The numeric core keeps its inherited model, independent of the caller's guest FP flags.
  if (int error =
          core((flags & RJ_ARITH_CPU_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
               exec_mask, 0, destinations, sources, nullptr)) {
    return error;
  }
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  const auto keep_destination = _mm256_set1_epi32(mode & RJ_ARITH_ALU_HIGH_D ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(output + lane));
    if (mode & RJ_ARITH_ALU_OMOD_MASK) {
      bits = _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(bits), scale));
      const auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
      bits = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), magnitude), bits);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      bits = _mm256_castps_si256(clamp_f32(_mm256_castsi256_ps(bits), false));
    }
    // Truncate, never RNE: AVX512BF16 narrowing would change the legacy numeric contract.
    bits = mode & RJ_ARITH_ALU_HIGH_D ? _mm256_and_si256(bits, _mm256_set1_epi32(-65536))
                                      : _mm256_srli_epi32(bits, 16);
    bits = _mm256_or_si256(
        bits, _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane)),
                               keep_destination));
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask, bits);
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace rocjitsu::arith
