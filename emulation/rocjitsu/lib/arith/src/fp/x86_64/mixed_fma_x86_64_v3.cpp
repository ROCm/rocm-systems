// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 mixed FP16/FP32 FMA with FP32 or selected-halfword output.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/mixed_fma.h"
#include "fp/mixed_fma_control.h"
#include "fp/scalar_repair.h"
#include "fp/x86_64/mixed_half_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Read eight mixed FP16/FP32 operands, applying half selection and source sign modifiers.
__m256 input8(const uint32_t *p, uint32_t mode) {
  auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  if (mode & RJ_ARITH_MIX_F16_A) {
    return rocjitsu::arith::half_input(words, mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, mode);
  }
  return _mm256_castsi256_ps(_mm256_xor_si256(
      _mm256_and_si256(words, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1)),
      _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0)));
}

// Read four mixed FP16/FP32 operands, applying half selection and source sign modifiers.
__m128 input4(const uint32_t *p, uint32_t mode) {
  auto words = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  if (mode & RJ_ARITH_MIX_F16_A) {
    auto halves =
        _mm_and_si128(_mm_srl_epi32(words, _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)),
                      _mm_set1_epi32(65535));
    words = _mm_castps_si128(_mm_cvtph_ps(_mm_packus_epi32(halves, halves)));
  }
  return _mm_castsi128_ps(_mm_xor_si128(
      _mm_and_si128(words, _mm_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1)),
      _mm_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0)));
}

} // namespace

namespace rocjitsu::arith {

namespace {

// Compute mixed-precision FMA into FP32 D with the selected guest policy; preserve inactive
// lanes.
template <bool GuestPolicy, bool Exact = false>
void mixed_float(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                 const uint32_t *b, const uint32_t *c, bool rdna3 = false) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = input8(a + lane, mode), y = input8(b + lane, mode >> 1),
         z = input8(c + lane, mode >> 2);
    if constexpr (GuestPolicy) {
      // Apply guest input flushing only to operands represented as FP32.
      auto flush = [flags](__m256 input, uint32_t source_mode) {
        if (!mixed_fma_flush_input(flags, source_mode)) {
          return input;
        }
        auto bits = _mm256_castps_si256(input);
        auto tiny = _mm256_cmpgt_epi32(
            _mm256_set1_epi32(source_mode & RJ_ARITH_MIX_F16_A ? 0x38800000 : 0x00800000),
            _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)));
        return _mm256_castsi256_ps(
            _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)), tiny));
      };
      x = flush(x, mode);
      y = flush(y, mode >> 1);
      z = flush(z, mode >> 2);
    }
    auto result = _mm256_fmadd_ps(x, y, z);
    unsigned repair = 0;
    if constexpr (Exact) {
      const auto magnitude = _mm256_set1_epi32(INT32_MAX);
      auto largest =
          _mm256_max_epi32(_mm256_and_si256(_mm256_castps_si256(x), magnitude),
                           _mm256_max_epi32(_mm256_and_si256(_mm256_castps_si256(y), magnitude),
                                            _mm256_and_si256(_mm256_castps_si256(z), magnitude)));
      auto m = _mm256_and_si256(_mm256_castps_si256(result), magnitude);
      auto bad =
          _mm256_or_si256(_mm256_cmpgt_epi32(largest, _mm256_set1_epi32(0x7f7fffff)),
                          _mm256_or_si256(_mm256_cmpgt_epi32(m, _mm256_set1_epi32(0x7f7fffff)),
                                          _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800001), m)));
      repair = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(bad))) & (exec_mask >> lane);
    }
    if (GuestPolicy && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      result = flush_denorm_f32(result);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      result = clamp_f32(result, GuestPolicy && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF));
    }
    if constexpr (Exact) {
      if (repair) {
        uint32_t out[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), _mm256_castps_si256(result));
        for (int i = 0; i < 8; ++i) {
          if (!((repair >> i) & 1)) {
            continue;
          }
          out[i] = repair_mixed_fma(MixedFma::Float, rdna3, flags, mode, a[lane + i], b[lane + i],
                                    c[lane + i], 0);
        }
        result = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(out)));
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(result));
  }
}

} // namespace

// Native FMA is exact for finite normal results under default host rounding.
// Tiny/nonfinite results retain lane-local scalar policy and NaN handling.
void mixed_fma_float_exact_x86_64_v3(bool rdna3, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                     uint32_t *d, const uint32_t *a, const uint32_t *b,
                                     const uint32_t *c) {
  mixed_float<true, true>(flags, exec_mask, mode, d, a, b, c, rdna3);
}

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_float_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c) {
  mixed_float<false>(0, exec_mask, mode, d, a, b, c);
}

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_float_guest_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  mixed_float<true>(flags, exec_mask, mode, d, a, b, c);
}

namespace {

// Compute mixed-precision FMA into one halfword of D, preserving the other half and inactive
// lanes.
template <bool GuestPolicy>
void mixed_half(bool high, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 4) {
    auto x = input4(a + lane, mode), y = input4(b + lane, mode >> 1),
         z = input4(c + lane, mode >> 2);
    if constexpr (GuestPolicy) {
      // Apply guest input flushing only to operands represented as FP32.
      auto flush = [flags](__m128 input, uint32_t source_mode) {
        if (!mixed_fma_flush_input(flags, source_mode)) {
          return input;
        }
        auto bits = _mm_castps_si128(input);
        auto tiny = _mm_cmplt_epi32(
            _mm_and_si128(bits, _mm_set1_epi32(INT32_MAX)),
            _mm_set1_epi32(source_mode & RJ_ARITH_MIX_F16_A ? 0x38800000 : 0x00800000));
        return _mm_castsi128_ps(
            _mm_blendv_epi8(bits, _mm_and_si128(bits, _mm_set1_epi32(INT32_MIN)), tiny));
      };
      x = flush(x, mode);
      y = flush(y, mode >> 1);
      z = flush(z, mode >> 2);
    }
    auto result =
        mixed_half_value4<false>(x, y, z, flags & RJ_ARITH_FP16_OVFL, mode & RJ_ARITH_ALU_CLAMP,
                                 GuestPolicy && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS),
                                 GuestPolicy && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF));
    unsigned shift = high ? 16 : 0;
    auto original = _mm_loadu_si128(reinterpret_cast<const __m128i *>(d + lane));
    auto output_mask = _mm_set1_epi32(int(uint32_t(65535) << shift));
    result = _mm_or_si128(_mm_sll_epi32(result, _mm_cvtsi32_si128(shift)),
                          _mm_andnot_si128(output_mask, original));
    auto lane_exec_mask =
        _mm_sllv_epi32(_mm_set1_epi32(int(exec_mask >> lane)), _mm_setr_epi32(31, 30, 29, 28));
    _mm_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_half_x86_64_v3(bool high, bool saturate, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c) {
  mixed_half<false>(high, saturate ? RJ_ARITH_FP16_OVFL : 0, exec_mask, mode, d, a, b, c);
}

// Evaluate mixed FP16/FP32 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void mixed_fma_half_guest_x86_64_v3(bool high, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                    uint32_t *d, const uint32_t *a, const uint32_t *b,
                                    const uint32_t *c) {
  mixed_half<true>(high, flags, exec_mask, mode, d, a, b, c);
}

} // namespace rocjitsu::arith
