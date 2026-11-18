// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP8/BF8 widening to FP32 or FP16, including packed pairs.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/fp8.h"
#include "conversion/fp8_conversion.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Widen FP8/BF8 factors to the selected FP32/FP16 destination format. Preserve destination lanes
// outside exec_mask. Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Packed, bool Half>
void fp8_conversion_x86_64_v4(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a, Fp8WidenPolicy policy, uint64_t flags,
                              uint32_t mode) {
  auto widen = [policy](__m512i value) {
    if (policy == Fp8WidenPolicy::E5M3) {
      return widen_fp8_e5m3(value);
    }
    if (policy == Fp8WidenPolicy::Gfx942) {
      return widen_fp8<Bf8, false, true>(value);
    }
    if (policy == Fp8WidenPolicy::Gfx950) {
      return widen_fp8<Bf8, false>(value);
    }
    return widen_fp8<Bf8, true>(value);
  };
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    raw = _mm512_srl_epi32(raw, _mm_cvtsi32_si128(int(shift)));
    auto low = _mm512_castps_si512(widen(raw));
    if constexpr (Half) {
      // All FP8/BF8 values fit FP16 exactly, except unsigned E5M3 finite overflow.
      // Decoded NaNs are quiet, so native conversion retains the inherited payload.
      auto half = [flags](__m512i value) {
        auto result = _mm512_cvtepu16_epi32(_mm512_cvt_roundps_ph(
            _mm512_castsi512_ps(value), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        if (flags & RJ_ARITH_FP16_OVFL) {
          auto magnitude = _mm512_and_si512(value, _mm512_set1_epi32(0x7fffffff));
          auto finite = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(0x7f800000), magnitude);
          auto overflow = _mm512_cmpeq_epi32_mask(
              _mm512_and_si512(result, _mm512_set1_epi32(0x7fff)), _mm512_set1_epi32(0x7c00));
          result = _mm512_mask_sub_epi32(result, finite & overflow, result, _mm512_set1_epi32(1));
        }
        return result;
      };

      low = half(low);
      if constexpr (Packed) {
        auto high = half(_mm512_castps_si512(widen(_mm512_srli_epi32(raw, 8))));
        low = _mm512_or_si512(low, _mm512_slli_epi32(high, 16));
      } else {
        auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d[0] + lane));
        bool high = mode & RJ_ARITH_ALU_HIGH_D;
        auto keep = _mm512_set1_epi32(high ? 0xffff : int(0xffff0000U));
        low = _mm512_or_si512(_mm512_and_si512(old, keep), high ? _mm512_slli_epi32(low, 16) : low);
      }
    }
    auto lane_exec_mask = __mmask16(exec_mask >> lane);
    if constexpr (Packed && !Half) {
      auto high = _mm512_castps_si512(widen(_mm512_srli_epi32(raw, 8)));
      _mm512_mask_storeu_epi32(d[0] + lane, lane_exec_mask, low);
      _mm512_mask_storeu_epi32(d[1] + lane, lane_exec_mask, high);
    } else {
      _mm512_mask_storeu_epi32(d[0] + lane, lane_exec_mask, low);
    }
  }
}

template void fp8_conversion_x86_64_v4<false, false, false>(uint32_t, unsigned, uint32_t *const *,
                                                            const uint32_t *, Fp8WidenPolicy,
                                                            uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<false, false, true>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<false, true, false>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<false, true, true>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<true, false, false>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<true, false, true>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<true, true, false>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v4<true, true, true>(uint32_t, unsigned, uint32_t *const *,
                                                         const uint32_t *, Fp8WidenPolicy, uint64_t,
                                                         uint32_t);

} // namespace rocjitsu::arith
