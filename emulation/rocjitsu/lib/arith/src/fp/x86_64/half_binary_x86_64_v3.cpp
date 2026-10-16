// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 add/subtract/multiply, including packed forms and guest FP status.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/packed_alu.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/binary.h"
#include "fp/half_binary.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP16 binary arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Binary Op, bool Packed>
void half_binary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, const uint32_t *b, bool compatibility) {
  constexpr bool arithmetic =
      Op == Binary::Add || Op == Binary::Sub || Op == Binary::Subrev || Op == Binary::Mul;

  uint32_t modes[] = {Packed ? packed_half_mode(mode, false) : mode, packed_half_mode(mode, true)};
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 8) {
    auto a_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto b_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto result = _mm256_setzero_si256();
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      uint32_t m = modes[half];
      auto x = half_input(a_words, m & RJ_ARITH_ALU_HIGH_A ? 16 : 0, m);
      auto y = half_input(b_words, m & RJ_ARITH_ALU_HIGH_B ? 16 : 0, m >> 1);
      if (arithmetic && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        // FP16 subnormals are normal FP32 values after conversion. Compare the
        // promoted magnitude against FP16's minimum normal, retaining the sign.
        auto flush_input = [](__m256 input) {
          auto bits = _mm256_castps_si256(input);
          auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
          auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude);
          auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN));
          return _mm256_castsi256_ps(_mm256_blendv_epi8(bits, sign, tiny));
        };
        x = flush_input(x);
        y = flush_input(y);
      }
      __m256 value;
      if constexpr (Op == Binary::Add) {
        value = _mm256_add_ps(x, y);
      }
      if constexpr (Op == Binary::Sub) {
        y = _mm256_xor_ps(y, _mm256_set1_ps(-0.0f));
        value = _mm256_add_ps(x, y);
      }
      if constexpr (Op == Binary::Subrev) {
        x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
        value = _mm256_add_ps(x, y);
      }
      if constexpr (Op == Binary::Mul) {
        value = _mm256_mul_ps(x, y);
      }

      if (compatibility) {
        // Preserve source priority and the explicit subtraction sign flip on NaNs.
        value = _mm256_blendv_ps(value, y, _mm256_cmp_ps(y, y, _CMP_UNORD_Q));
        value = _mm256_blendv_ps(value, x, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
      }
      if (m & RJ_ARITH_ALU_OMOD_MASK) {
        if (compatibility && !Packed) {
          // Narrow and widen in registers; ordinary Loose needs neither conversion.
          auto narrowed = half_narrow(value, flags & RJ_ARITH_FP16_OVFL);
          value = _mm256_cvtph_ps(_mm_packus_epi32(_mm256_castsi256_si128(narrowed),
                                                   _mm256_extracti128_si256(narrowed, 1)));
        }
        value = prepare_omod_f16(value, m);
        value = _mm256_mul_ps(value, _mm256_set1_ps(scales[(m >> 6) & 3]));
      }
      if (m & RJ_ARITH_ALU_CLAMP) {
        auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
        if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
          auto magnitude =
              _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX));
          auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
          clamped = _mm256_blendv_ps(clamped, value, _mm256_castsi256_ps(nan));
        }
        value = clamped;
      }
      auto encoded = half_narrow(value, flags & RJ_ARITH_FP16_OVFL);
      if (arithmetic && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        auto magnitude = _mm256_and_si256(encoded, _mm256_set1_epi32(0x7fff));
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x0400), magnitude);
        auto sign = _mm256_and_si256(encoded, _mm256_set1_epi32(0x8000));
        encoded = _mm256_blendv_epi8(encoded, sign, tiny);
      }
      if constexpr (Packed) {
        result = _mm256_or_si256(result, _mm256_sll_epi32(encoded, _mm_cvtsi32_si128(16 * half)));
      } else {
        result = _mm256_sll_epi32(encoded, _mm_cvtsi32_si128(d_shift));
      }
    }
    if constexpr (!Packed) {
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void half_binary_x86_64_v3<Binary::Add>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *, bool);
template void half_binary_x86_64_v3<Binary::Sub>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *, bool);
template void half_binary_x86_64_v3<Binary::Subrev>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *, bool);
template void half_binary_x86_64_v3<Binary::Mul>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *, bool);

template void half_binary_x86_64_v3<Binary::Add, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *, bool);
template void half_binary_x86_64_v3<Binary::Mul, true>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *, bool);

} // namespace rocjitsu::arith
