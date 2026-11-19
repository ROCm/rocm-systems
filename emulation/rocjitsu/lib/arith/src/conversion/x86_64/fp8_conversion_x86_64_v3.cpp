// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP8/BF8 widening to FP32 or FP16, including packed pairs.
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
void fp8_conversion_x86_64_v3(uint32_t exec_mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a, Fp8WidenPolicy policy, uint64_t flags,
                              uint32_t mode) {
  auto widen = [policy](__m256i value) {
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
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    raw = _mm256_srl_epi32(raw, _mm_cvtsi32_si128(int(shift)));
    auto low = _mm256_castps_si256(widen(raw));
    if constexpr (Half) {
      // All FP8/BF8 values fit FP16 exactly, except unsigned E5M3 finite overflow.
      // Decoded NaNs are quiet, so native conversion retains the inherited payload.
      auto half = [flags](__m256i value) {
        auto result = _mm256_cvtepu16_epi32(_mm256_cvtps_ph(
            _mm256_castsi256_ps(value), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        if (flags & RJ_ARITH_FP16_OVFL) {
          auto magnitude = _mm256_and_si256(value, _mm256_set1_epi32(0x7fffffff));
          auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
          auto overflow = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                             _mm256_set1_epi32(0x7c00));
          result = _mm256_blendv_epi8(result, _mm256_sub_epi32(result, _mm256_set1_epi32(1)),
                                      _mm256_and_si256(finite, overflow));
        }
        return result;
      };

      low = half(low);
      if constexpr (Packed) {
        auto high = half(_mm256_castps_si256(widen(_mm256_srli_epi32(raw, 8))));
        low = _mm256_or_si256(low, _mm256_slli_epi32(high, 16));
      } else {
        auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[0] + lane));
        bool high = mode & RJ_ARITH_ALU_HIGH_D;
        auto keep = _mm256_set1_epi32(high ? 0xffff : int(0xffff0000U));
        low = _mm256_or_si256(_mm256_and_si256(old, keep), high ? _mm256_slli_epi32(low, 16) : low);
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    if constexpr (Packed && !Half) {
      auto high = _mm256_castps_si256(widen(_mm256_srli_epi32(raw, 8)));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), lane_exec_mask, low);
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[1] + lane), lane_exec_mask, high);
    } else {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), lane_exec_mask, low);
    }
  }
}

template void fp8_conversion_x86_64_v3<false, false, false>(uint32_t, unsigned, uint32_t *const *,
                                                            const uint32_t *, Fp8WidenPolicy,
                                                            uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<false, false, true>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<false, true, false>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<false, true, true>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<true, false, false>(uint32_t, unsigned, uint32_t *const *,
                                                           const uint32_t *, Fp8WidenPolicy,
                                                           uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<true, false, true>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<true, true, false>(uint32_t, unsigned, uint32_t *const *,
                                                          const uint32_t *, Fp8WidenPolicy,
                                                          uint64_t, uint32_t);

template void fp8_conversion_x86_64_v3<true, true, true>(uint32_t, unsigned, uint32_t *const *,
                                                         const uint32_t *, Fp8WidenPolicy, uint64_t,
                                                         uint32_t);

} // namespace rocjitsu::arith
