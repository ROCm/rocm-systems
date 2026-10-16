// SPDX-License-Identifier: MIT

// AVX-512 exact FP32 integral rounding and guest FP status classification.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/unary.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Round FP32 inputs to integral values with the selected rounding code and guest output
// controls.
template <int Rounding>
void round(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a, uint64_t flags) {
  auto keep = _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto flip = _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 16) {
    auto bits = _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(a + lane), keep), flip);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800000));
      bits =
          _mm512_mask_mov_epi32(bits, tiny, _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MIN)));
    }
    // Instruction-local rounding, never an MXCSR change. NaNs are quieted
    // without losing their sign/payload, as in the inherited integer model.
    auto value = _mm512_roundscale_ps(_mm512_castsi512_ps(bits), Rounding | _MM_FROUND_NO_EXC);
    if (mode & RJ_ARITH_ALU_OMOD_HALF) {
      // Integral results cannot be subnormal. OMOD maps either zero sign to +0.
      auto zero = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ);
      value = _mm512_mask_mov_ps(value, zero, _mm512_setzero_ps());
      value = _mm512_mul_ps(value, _mm512_set1_ps(scales[(mode >> 6) & 3]));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        value = _mm512_min_ps(_mm512_set1_ps(1), _mm512_max_ps(_mm512_setzero_ps(), value));
        auto zero = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ);
        value = _mm512_mask_mov_ps(value, zero, _mm512_setzero_ps());
      } else {
        value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
      }
    }
    _mm512_mask_storeu_ps(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

} // namespace

// Evaluate round exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void round_exact_x86_64_v4(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags) {
  switch (op) {
  case Unary::Trunc:
    return round<_MM_FROUND_TO_ZERO>(exec_mask, mode, d, a, flags);
  case Unary::Ceil:
    return round<_MM_FROUND_TO_POS_INF>(exec_mask, mode, d, a, flags);
  case Unary::Floor:
    return round<_MM_FROUND_TO_NEG_INF>(exec_mask, mode, d, a, flags);
  case Unary::Rndne:
    return round<_MM_FROUND_TO_NEAREST_INT>(exec_mask, mode, d, a, flags);
  default:
    __builtin_unreachable();
  }
}

} // namespace rocjitsu::arith
