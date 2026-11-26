// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP32 mantissa normalization for all input encodings.
// Guest controls and NaN policy match the inherited scalar models.
#include "common/x86_64/alu_x86_64_v4.h"
#include "math/frexp_mant.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Stage normalized mantissas before stores, including register aliases.
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
         const uint32_t *const *a, bool quiet_snan) {
  uint32_t result[1][32];
  for (int lane = 0; lane < 32; lane += 16) {

    auto bits = _mm512_loadu_si512(a[0] + lane);
    bits = _mm512_and_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
    bits = _mm512_xor_si512(bits, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
    auto source = _mm512_castsi512_ps(bits);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      source = flush_denorm_f32(source);
    }
    bits = _mm512_castps_si512(source);
    auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
    auto finite_nonzero = _mm512_cmpneq_epi32_mask(magnitude, _mm512_setzero_si512()) &
                          _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto value = _mm512_mask_getmant_ps(source, finite_nonzero, source, _MM_MANT_NORM_p5_1,
                                        _MM_MANT_SIGN_src);
    auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto encoded = _mm512_castps_si512(value);
    if (quiet_snan) {
      encoded = _mm512_mask_or_epi32(encoded, nan, encoded, _mm512_set1_epi32(0x00400000));
    }
    const unsigned omod = (mode >> 6) & 3;
    if (omod) {
      // A normalized mantissa remains normal after every OMOD; scaling is exact.
      const int delta = omod == 3 ? -1 : int(omod);
      encoded = _mm512_mask_add_epi32(encoded, finite_nonzero, encoded,
                                      _mm512_set1_epi32(delta * 0x00800000));
      encoded =
          _mm512_mask_mov_epi32(encoded, _mm512_cmpeq_epi32_mask(magnitude, _mm512_setzero_si512()),
                                _mm512_setzero_si512());
    }
    value = _mm512_castsi512_ps(encoded);
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    // The gfx11 policy may return a signaling NaN without quieting, including under DX9 CLAMP.
    if (!quiet_snan && (!(mode & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
      value = _mm512_mask_mov_ps(value, nan, _mm512_castsi512_ps(bits));
    }
    _mm512_storeu_si512(result[0] + lane, _mm512_castps_si512(value));
  }
  for (int word = 0; word < 1; ++word) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[word] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[word] + lane));
    }
  }
}

} // namespace

// Normalize the FP32 source with validated guest and CPU controls.
void frexp_mant_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, bool quiet_snan) {
  run(flags, exec_mask, mode, d, a, quiet_snan);
}

} // namespace rocjitsu::arith
