// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 exact FP16 special-function execution using SIMD with scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "math/half_unary.h"
#include "math/x86_64/rcp_exact.h"
#include "math/x86_64/root_exact.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Widen exactly, reuse the inherited FP32 mapping, then apply half-width
// narrowing/OMOD. Conversion preserves signaling payloads; only the mapping
// decides whether to quiet them. Staging permits D to alias the selected half.
template <Unary Op>
void half_sfu_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, bool quiet_snan) {
  uint32_t input[32], output[32];
  const auto sign = _mm256_set1_epi32(0x80000000U);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<true>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                              mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
    auto bits = _mm256_castps_si256(x);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny =
          _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), _mm256_andnot_si256(sign, bits));
      bits = _mm256_blendv_epi8(bits, _mm256_and_si256(bits, sign), tiny);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(input + lane), bits);
  }
  // Guest half denormals have already been handled. Wide results must not be
  // clamped/scaled/flushed until they have been narrowed to the half format.
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if constexpr (Op == Unary::Rcp) {
      rcp_exact_x86_64_v4(0, 0xffffffffU, 0, output, input, quiet_snan);
    }

    // Evaluate root exact for the caller-selected operation.
    // Requires the named CPU features and validated instruction controls.
    else {
      root_exact_x86_64_v4(Op == Unary::Rsq, 0, 0xffffffffU, 0, output, input, quiet_snan, false);
    }
  } else
#endif
  {
    if constexpr (Op == Unary::Rcp) {
      rcp_exact_x86_64_v3(0, 0xffffffffU, 0, output, input, quiet_snan);
    }

    // Evaluate root exact for the caller-selected operation.
    // Requires the named CPU features and validated instruction controls.
    else {
      root_exact_x86_64_v3(Op == Unary::Rsq, 0, 0xffffffffU, 0, output, input, quiet_snan, false);
    }
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto wide = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(output + lane));
    auto value = half_narrow<true>(_mm256_castsi256_ps(wide), flags & RJ_ARITH_FP16_OVFL);
    if (flags & RJ_ARITH_FP16_OVFL) {
      auto infinity =
          _mm256_cmpeq_epi32(_mm256_andnot_si256(sign, wide), _mm256_set1_epi32(0x7f800000));
      auto finite = _mm256_cmpgt_epi32(
          _mm256_set1_epi32(0x7f800000),
          _mm256_andnot_si256(sign,
                              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + lane))));
      value = _mm256_add_epi32(value, _mm256_and_si256(infinity, finite));
    }
    auto magnitude = _mm256_and_si256(value, _mm256_set1_epi32(0x7fff));
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude);
      value = _mm256_blendv_epi8(value, _mm256_and_si256(value, _mm256_set1_epi32(0x8000)), tiny);
      magnitude = _mm256_and_si256(value, _mm256_set1_epi32(0x7fff));
    }
    if (unsigned omod = (mode >> 6) & 3) {
      auto scaled =
          _mm256_add_epi32(value, _mm256_set1_epi32(omod == 3 ? -0x400 : int(omod << 10)));
      if (omod != 3) {
        auto overflow = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7bff - (omod << 10)));
        auto limit =
            _mm256_or_si256(_mm256_and_si256(value, _mm256_set1_epi32(0x8000)),
                            _mm256_set1_epi32(flags & RJ_ARITH_FP16_OVFL ? 0x7bff : 0x7c00));
        scaled = _mm256_blendv_epi8(scaled, limit, overflow);
      }
      if constexpr (Op == Unary::Rcp) {
        if (omod == 3) {
          scaled = _mm256_blendv_epi8(scaled, _mm256_and_si256(value, _mm256_set1_epi32(0x8000)),
                                      _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800), magnitude));
        }
      }
      auto zero = Op == Unary::Rcp ? _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude)
                                   : _mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256());
      scaled = _mm256_andnot_si256(zero, scaled);
      value = _mm256_blendv_epi8(value, scaled,
                                 _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7c00), magnitude));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto magnitude = _mm256_and_si256(value, _mm256_set1_epi32(0x7fff));
      auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7c00));
      auto negative = _mm256_cmpgt_epi32(value, _mm256_set1_epi32(0x7fff));
      auto clamped = _mm256_andnot_si256(_mm256_or_si256(nan, negative),
                                         _mm256_min_epu32(value, _mm256_set1_epi32(0x3c00)));
      value =
          flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? _mm256_blendv_epi8(clamped, value, nan) : clamped;
    }
    const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    value = _mm256_or_si256(_mm256_sll_epi32(value, _mm_cvtsi32_si128(shift)),
                            _mm256_and_si256(old, _mm256_set1_epi32(shift ? 0xffff : 0xffff0000U)));
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask, value);
  }
}

template void half_sfu_exact_x86_64_v3<Unary::Rcp>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                   const uint32_t *, bool);
template void half_sfu_exact_x86_64_v3<Unary::Rsq>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                   const uint32_t *, bool);
template void half_sfu_exact_x86_64_v3<Unary::Sqrt>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, bool);

} // namespace rocjitsu::arith
