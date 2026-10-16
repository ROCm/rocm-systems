// SPDX-License-Identifier: MIT

// AVX-512 FP32 scaling by powers of two with guest rounding and denormal policies.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/ldexp.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Scale FP32 by powers of two for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <bool Fp64, bool GuestDenormals>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
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
      if (mode & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MAX));
      }
      if (mode & RJ_ARITH_ALU_NEG_A) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi64(INT64_MIN));
      }
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        bits = _mm512_castpd_si512(flush_denorm_f64(_mm512_castsi512_pd(bits)));
      }
      auto exponent =
          _mm512_cvtepi32_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)));
      auto value = _mm512_scalef_pd(_mm512_castsi512_pd(bits), exponent);
      if (mode & RJ_ARITH_ALU_OMOD_HALF) {
        auto exact_exponent = _mm512_add_pd(_mm512_getexp_pd(_mm512_castsi512_pd(bits)), exponent);
        auto tiny_before_rounding =
            _mm512_cmp_pd_mask(exact_exponent, _mm512_set1_pd(-1022), _CMP_LT_OQ);
        value = _mm512_mask_mov_pd(value, tiny_before_rounding, _mm512_setzero_pd());
        value = prepare_omod_f64(value, mode);
        value = _mm512_mul_pd(value, _mm512_set1_pd(scales[(mode >> 6) & 3]));
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
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
      if (mode & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      }
      if (mode & RJ_ARITH_ALU_NEG_A) {
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
      if (mode & RJ_ARITH_ALU_OMOD_HALF) {
        auto exact_exponent = _mm512_add_ps(_mm512_getexp_ps(_mm512_castsi512_ps(bits)),
                                            _mm512_cvtepi32_ps(exponent));
        auto tiny_before_rounding =
            _mm512_cmp_ps_mask(exact_exponent, _mm512_set1_ps(-126), _CMP_LT_OQ);
        value = _mm512_mask_mov_ps(value, tiny_before_rounding, _mm512_setzero_ps());
        value = prepare_omod_f32(value, mode);
        value = _mm512_mul_ps(value, _mm512_set1_ps(scales[(mode >> 6) & 3]));
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
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

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b) {
  if (fp64) {
    run<true, false>(0, exec_mask, mode, d, a, b);
  } else {
    run<false, false>(0, exec_mask, mode, d, a, b);
  }
}

// Scale FP32 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_guest_x86_64_v4(bool fp64, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *b) {
  if (fp64) {
    run<true, true>(flags, exec_mask, mode, d, a, b);
  } else {
    run<false, true>(flags, exec_mask, mode, d, a, b);
  }
}

} // namespace rocjitsu::arith
