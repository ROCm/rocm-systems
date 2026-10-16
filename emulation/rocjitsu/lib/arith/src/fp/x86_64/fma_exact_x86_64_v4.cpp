// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 exact FMA fast paths using instruction-local rounding and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/x86_64/fma_exact_x86_64_v4.h"
#include "common/internal.h"
#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>

namespace rocjitsu::arith {

namespace {

// Keep the inherited scalar machinery out of the normal SIMD loop.
[[gnu::noinline]] __m512i repair_fma(__m512i result, __mmask16 bad, uint64_t flags, uint32_t mode,
                                     const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                     bool rdna3, bool dx9_zero) {
  uint32_t repaired[16];
  _mm512_storeu_si512(repaired, result);
  for (int i = 0; i < 16; ++i) {
    if ((bad >> i) & 1) {
      repaired[i] = repair_fma_f32(a[i], b[i], c[i], flags, mode, rdna3, dx9_zero);
    }
  }
  return _mm512_loadu_si512(repaired);
}

} // namespace

// In the admitted normal-or-exact-zero domain, locally rounded FMA matches the
// inherited rocjitsu model. Only special/tiny lanes use scalar repair.
// Stage both halves before stores to preserve aliases during repair.
bool fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c, bool dx9_zero,
                         bool rdna3) {
  // Stage both vector chunks before destination writes; repaired lanes must still see original
  // inputs.
  __m512i out[2];
  for (int k = 0; k < 2; ++k) {
    __mmask16 bad = 0;
    __m512 aa = prepare_fma_input(_mm512_loadu_si512(a + 16 * k), flags, mode, 0, bad);
    __m512 bb = prepare_fma_input(_mm512_loadu_si512(b + 16 * k), flags, mode, 1, bad);
    __m512 cc = prepare_fma_input(_mm512_loadu_si512(c + 16 * k), flags, mode, 2, bad);
    if (dx9_zero) {
      const auto zero = _mm512_setzero_ps();
      auto has_zero =
          _mm512_cmp_ps_mask(aa, zero, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(bb, zero, _CMP_EQ_OQ);
      aa = _mm512_mask_mov_ps(aa, has_zero, zero);
      bb = _mm512_mask_mov_ps(bb, has_zero, _mm512_set1_ps(1));
    }
    out[k] = fma_result(aa, bb, cc, flags, mode, bad);
    if (bad) {
      out[k] =
          repair_fma(out[k], bad, flags, mode, a + 16 * k, b + 16 * k, c + 16 * k, rdna3, dx9_zero);
    }
  }
  _mm512_mask_storeu_epi32(d, __mmask16(exec_mask), out[0]);
  _mm512_mask_storeu_epi32(d + 16, __mmask16(exec_mask >> 16), out[1]);
  return true;
}

} // namespace rocjitsu::arith
