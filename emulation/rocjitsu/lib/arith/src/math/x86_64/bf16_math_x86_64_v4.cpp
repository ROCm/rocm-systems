// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 BF16 math packing and modifiers around existing FP32 numeric backends.
#include "common/x86_64/alu_x86_64_v4.h"
#include "math/x86_64/bf16_math.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Stage the entire source before writing, preserving aliases, inactive lanes and other halves.
int bf16_math_x86_64_v4(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a) {
  uint32_t input[32], output[32]{};
  const auto zero = _mm512_setzero_si512();
  const auto keep_source = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : -1);
  const auto flip = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? int(0x80000000U) : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto bits = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm512_and_si512(bits, _mm512_set1_epi32(-65536))
                                      : _mm512_slli_epi32(bits, 16);
    bits = _mm512_xor_si512(_mm512_and_si512(bits, keep_source), flip);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(input + lane), bits);
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
  const auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  const auto keep_destination = _mm512_set1_epi32(mode & RJ_ARITH_ALU_HIGH_D ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 16) {
    auto bits = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(output + lane));
    if (mode & RJ_ARITH_ALU_OMOD_MASK) {
      bits = _mm512_castps_si512(_mm512_mul_ps(_mm512_castsi512_ps(bits), scale));
      const auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
      bits = _mm512_mask_mov_epi32(
          bits, _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800000)), zero);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      bits = _mm512_castps_si512(clamp_f32(_mm512_castsi512_ps(bits), false));
    }
    // Truncate, never RNE: AVX512BF16 narrowing would change the legacy numeric contract.
    bits = mode & RJ_ARITH_ALU_HIGH_D ? _mm512_and_si512(bits, _mm512_set1_epi32(-65536))
                                      : _mm512_srli_epi32(bits, 16);
    bits = _mm512_or_si512(
        bits, _mm512_and_si512(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane)),
                               keep_destination));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), bits);
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace rocjitsu::arith
