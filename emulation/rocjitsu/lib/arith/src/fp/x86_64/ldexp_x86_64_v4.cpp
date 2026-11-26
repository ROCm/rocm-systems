// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/bits.h"
#include "common/compiler.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/ldexp.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Scale FP32 by powers of two for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool Fp64, bool GuestDenormals>
void run(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *b) {
  uint32_t result[Fp64 ? 2 : 1][32];
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += Fp64 ? 8 : 16) {
    if constexpr (Fp64) {
      auto low =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
      auto high =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)));
      auto bits = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
      if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MAX));
      }
      if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi64(INT64_MIN));
      }
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        bits = _mm512_castpd_si512(flush_denorm_f64(_mm512_castsi512_pd(bits)));
      }
      auto exponent =
          _mm512_cvtepi32_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)));
      auto value = _mm512_scalef_pd(_mm512_castsi512_pd(bits), exponent);
      if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) ||
          (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
        auto exact_exponent = _mm512_add_pd(_mm512_getexp_pd(_mm512_castsi512_pd(bits)), exponent);
        auto tiny_before_rounding =
            _mm512_cmp_pd_mask(exact_exponent, _mm512_set1_pd(-1022), _CMP_LT_OQ);
        auto zero = instruction_flags & RJ_ARITH_ALU_OMOD_HALF
                        ? _mm512_setzero_pd()
                        : _mm512_castsi512_pd(_mm512_and_si512(bits, _mm512_set1_epi64(INT64_MIN)));
        value = _mm512_mask_mov_pd(value, tiny_before_rounding, zero);
      }
      if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
        value = prepare_omod_f64(value, instruction_flags);
        value = _mm512_mul_pd(value, _mm512_set1_pd(scales[(instruction_flags >> 6) & 3]));
      }
      if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
        auto clamped = _mm512_min_pd(_mm512_max_pd(value, _mm512_setzero_pd()), _mm512_set1_pd(1));
        if (GuestDenormals && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          clamped =
              _mm512_mask_mov_pd(clamped, _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q), value);
        }
        value = clamped;
      }
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        value = flush_denorm_f64(value);
      }
      bits = _mm512_castpd_si512(value);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                          _mm512_cvtepi64_epi32(bits));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
    } else {
      auto bits = _mm512_loadu_si512(a[0] + lane);
      if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      }
      if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi32(INT32_MIN));
      }
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        bits = _mm512_castps_si512(flush_denorm_f32(_mm512_castsi512_ps(bits)));
      }
      auto exponent = _mm512_loadu_si512(b + lane);

      // Exponents beyond these bounds have the same overflow/underflow result;
      // bounding them also makes the integer-to-float conversion exact.
      exponent = _mm512_min_epi32(_mm512_max_epi32(exponent, _mm512_set1_epi32(-4096)),
                                  _mm512_set1_epi32(4096));
      auto value = _mm512_scalef_ps(_mm512_castsi512_ps(bits), _mm512_cvtepi32_ps(exponent));
      if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) ||
          (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
        auto exact_exponent = _mm512_add_ps(_mm512_getexp_ps(_mm512_castsi512_ps(bits)),
                                            _mm512_cvtepi32_ps(exponent));
        auto tiny_before_rounding =
            _mm512_cmp_ps_mask(exact_exponent, _mm512_set1_ps(-126), _CMP_LT_OQ);
        auto zero = instruction_flags & RJ_ARITH_ALU_OMOD_HALF
                        ? _mm512_setzero_ps()
                        : _mm512_castsi512_ps(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MIN)));
        value = _mm512_mask_mov_ps(value, tiny_before_rounding, zero);
      }
      if (instruction_flags & RJ_ARITH_ALU_OMOD_HALF) {
        value = prepare_omod_f32(value, instruction_flags);
        value = _mm512_mul_ps(value, _mm512_set1_ps(scales[(instruction_flags >> 6) & 3]));
      }
      if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
        auto clamped = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
        if (GuestDenormals && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          clamped =
              _mm512_mask_mov_ps(clamped, _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q), value);
        }
        value = clamped;
      }
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        value = flush_denorm_f32(value);
      }
      _mm512_storeu_si512(result[0] + lane, _mm512_castps_si512(value));
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

} // namespace

// Stage exact normal scales and preserved/quieted NaNs. Return only the lanes
// whose subnormal input or rounded result needs the baseline integer helper.
uint32_t ldexp_f64_normal_x86_64_v4(uint64_t flags, uint32_t instruction_flags, bool quiet_snan,
                                    uint32_t result[2][32], const uint32_t *const *a,
                                    const uint32_t *b) {
  uint32_t scalar_lanes = 0;
  const auto zero = _mm512_setzero_si512(), maximum = _mm512_set1_epi64(2047);
  RJ_ARITH_UNROLL(4)
  for (int lane = 0; lane < 32; lane += 8) {
    auto low =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
    auto high =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)));
    auto raw = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      raw = _mm512_and_si512(raw, _mm512_set1_epi64(INT64_MAX));
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      raw = _mm512_xor_si512(raw, _mm512_set1_epi64(INT64_MIN));
    }
    auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi64(INT64_MAX));
    auto exponent = _mm512_srli_epi64(magnitude, 52);
    auto adjustment =
        _mm512_cvtepi32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)));
    adjustment = _mm512_min_epi64(_mm512_max_epi64(adjustment, _mm512_set1_epi64(-4096)),
                                  _mm512_set1_epi64(4096));
    auto scaled_exponent = _mm512_add_epi64(exponent, adjustment);
    auto normal = _mm512_cmp_epi64_mask(exponent, zero, _MM_CMPINT_GT) &
                  _mm512_cmp_epi64_mask(exponent, maximum, _MM_CMPINT_LT) &
                  _mm512_cmp_epi64_mask(scaled_exponent, zero, _MM_CMPINT_GT) &
                  _mm512_cmp_epi64_mask(scaled_exponent, maximum, _MM_CMPINT_LT);
    auto bits =
        _mm512_or_si512(_mm512_and_si512(raw, _mm512_set1_epi64(INT64_MIN | 0x000fffffffffffffLL)),
                        _mm512_slli_epi64(scaled_exponent, 52));
    bits = _mm512_mask_mov_epi64(raw, normal, bits);
    if (quiet_snan) {
      bits = _mm512_mask_or_epi64(
          bits,
          _mm512_cmp_epi64_mask(magnitude, _mm512_set1_epi64(0x7ff0000000000000LL), _MM_CMPINT_GT),
          bits, _mm512_set1_epi64(0x0008000000000000LL));
    }
    auto resolved = normal | _mm512_cmp_epi64_mask(magnitude, zero, _MM_CMPINT_EQ) |
                    _mm512_cmp_epi64_mask(exponent, maximum, _MM_CMPINT_EQ);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto tiny = _mm512_cmp_epi64_mask(exponent, zero, _MM_CMPINT_EQ);
      bits = _mm512_mask_mov_epi64(bits, tiny, _mm512_and_si512(raw, _mm512_set1_epi64(INT64_MIN)));
      resolved |= tiny;
    }
    scalar_lanes |= uint32_t(~resolved & 255) << lane;
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), _mm512_cvtepi64_epi32(bits));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
  }
  return scalar_lanes;
}

// Publish staged words only after every source and scalar repair has been read.
void ldexp_f64_store_x86_64_v4(uint32_t exec_mask, uint32_t *const *d,
                               const uint32_t result[2][32]) {
  RJ_ARITH_UNROLL(2)
  for (int reg = 0; reg < 2; ++reg) {
    RJ_ARITH_UNROLL(2)
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b) {
  if (fp64) {
    run<true, false>(0, exec_mask, instruction_flags, d, a, b);
  } else {
    run<false, false>(0, exec_mask, instruction_flags, d, a, b);
  }
}

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_guest_x86_64_v4(bool fp64, uint64_t flags, uint32_t exec_mask,
                           uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *b) {
  if (fp64) {
    run<true, true>(flags, exec_mask, instruction_flags, d, a, b);
  } else {
    run<false, true>(flags, exec_mask, instruction_flags, d, a, b);
  }
}

// AVX-512 exact scaling shares the baseline model only for boundary/subnormal
// lanes. Source aliases are safe because publication follows every repair.
void ldexp_f32_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                               uint32_t *d, const uint32_t *a, const uint32_t *b, bool quiet_snan) {
  uint32_t result[32];
  uint32_t repairs = 0;
  const unsigned omod = (instruction_flags >> 6) & 3;
  const auto delta = _mm512_set1_epi32(omod == 3 ? -1 : int(omod));
  const auto zero = _mm512_setzero_si512(), infinity = _mm512_set1_epi32(0x7f800000);
  RJ_ARITH_UNROLL(2)
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(a + lane);
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      raw = _mm512_and_si512(raw, _mm512_set1_epi32(INT32_MAX));
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      raw = _mm512_xor_si512(raw, _mm512_set1_epi32(INT32_MIN));
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      raw = _mm512_castps_si512(flush_denorm_f32(_mm512_castsi512_ps(raw)));
    }
    const auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(INT32_MAX));
    const auto exponent = _mm512_srli_epi32(magnitude, 23);
    auto adjustment = _mm512_loadu_si512(b + lane);
    adjustment = _mm512_min_epi32(_mm512_max_epi32(adjustment, _mm512_set1_epi32(-4096)),
                                  _mm512_set1_epi32(4096));
    const auto scaled = _mm512_add_epi32(exponent, adjustment);
    const auto final_exponent = _mm512_add_epi32(scaled, delta);
    // Keep original and OMOD ranges separate, preserving intermediate flushing.
    const __mmask16 normal = _mm512_cmpgt_epi32_mask(exponent, zero) &
                             _mm512_cmplt_epi32_mask(exponent, _mm512_set1_epi32(255)) &
                             _mm512_cmpgt_epi32_mask(scaled, zero) &
                             _mm512_cmplt_epi32_mask(scaled, _mm512_set1_epi32(255)) &
                             _mm512_cmpgt_epi32_mask(final_exponent, zero) &
                             _mm512_cmplt_epi32_mask(final_exponent, _mm512_set1_epi32(255));
    const __mmask16 zeros = _mm512_cmpeq_epi32_mask(magnitude, zero);
    const __mmask16 special = zeros | _mm512_cmpge_epi32_mask(magnitude, infinity);
    const __mmask16 nan = _mm512_cmpgt_epi32_mask(magnitude, infinity);
    auto special_value =
        _mm512_mask_or_epi32(raw, nan, raw, _mm512_set1_epi32(quiet_snan ? 0x00400000 : 0));
    if (omod) {
      special_value = _mm512_mask_mov_epi32(special_value, zeros, zero);
    }
    auto value = _mm512_or_si512(_mm512_and_si512(raw, _mm512_set1_epi32(INT32_MIN | 0x7fffff)),
                                 _mm512_slli_epi32(final_exponent, 23));
    value = _mm512_mask_mov_epi32(value, special, special_value);
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      value = _mm512_castps_si512(
          clamp_f32(_mm512_castsi512_ps(value), flags & RJ_ARITH_FP_DX10_CLAMP_OFF));
    }
    _mm512_storeu_si512(result + lane, value);
    repairs |= uint32_t(uint16_t(~(normal | special))) << lane;
  }
  while (repairs) {
    const int lane = int(bit_count_zero<true>(repairs));
    repairs &= repairs - 1;
    result[lane] = ldexp_f32_value_baseline(a[lane], bit_cast<int32_t>(b[lane]), flags,
                                            instruction_flags, quiet_snan);
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
}

} // namespace rocjitsu::arith
