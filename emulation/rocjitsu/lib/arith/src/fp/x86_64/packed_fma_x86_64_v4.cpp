// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 Loose packed FP16 FMA/FMAC with independent low/high operand selectors.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/packed_alu.h"
#include "common/sdwa_output.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "conversion/x86_64/lanes.h"
#include "fp/packed_fma.h"
#include "fp/x86_64/loose_half_output_x86_64_v4.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

namespace {

// Evaluate one packed FP16 FMA result half under the supplied per-half selectors and guest
// controls.
__m512i compute_half(__m512i a, __m512i b, __m512i c, uint64_t flags, uint32_t mode) {
  const auto input = [flags](__m512i raw, uint32_t controls) {
    auto value = half_input(raw, controls & RJ_ARITH_ALU_HIGH_A ? 16 : 0, controls);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      const auto bits = _mm512_castps_si512(value);
      const auto tiny = _mm512_cmplt_epi32_mask(
          _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX)), _mm512_set1_epi32(0x38800000));
      value = _mm512_castsi512_ps(
          _mm512_mask_and_epi32(bits, tiny, bits, _mm512_set1_epi32(INT32_MIN)));
    }
    return value;
  };
  const auto x = input(a, mode), y = input(b, mode >> 1), z = input(c, mode >> 2);
  return loose_half_output(_mm512_fmadd_ps(x, y, z), flags, mode);
}

} // namespace

// Evaluate paired FP16 fused multiply-adds. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Accumulate>
void packed_fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  const auto output = sdwa_output_policy(flags, mode);
  uint32_t low_mode, high_mode;
  if constexpr (Accumulate) {
    low_mode = mode & ~RJ_ARITH_ALU_SDWA;
    high_mode = low_mode | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C;
  } else {
    low_mode = packed_half_instruction_flags(mode, false);
    high_mode = packed_half_instruction_flags(mode, true);
  }
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(a + lane);
    auto vb = _mm512_loadu_si512(b + lane);
    auto vc = _mm512_loadu_si512(c + lane);
    auto low = compute_half(va, vb, vc, flags, low_mode);
    auto high = compute_half(va, vb, vc, flags, high_mode);
    auto result = _mm512_or_si512(low, _mm512_slli_epi32(high, 16));
    if constexpr (Accumulate) {
      if (mode & RJ_ARITH_ALU_SDWA) {
        result = sdwa_finalize_packed_f16(conversion_simd::Lanes<32>(result), output).bits;
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void packed_fma_x86_64_v4<false>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                          const uint32_t *, const uint32_t *, const uint32_t *);
template void packed_fma_x86_64_v4<true>(uint64_t, uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                         const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
