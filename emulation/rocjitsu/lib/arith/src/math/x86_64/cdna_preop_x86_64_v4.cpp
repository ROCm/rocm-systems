// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX512 compact CDNA range-reduction lookup and integer FP64 encoding.
#include "common/isa.h"
#include "math/cdna_trig_preop.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
using U = uint64_t __attribute__((vector_size(64)));
// Widen unsigned VGPR words to arithmetic lanes.
U load_words(const uint32_t *p) {
  return U(_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p))));
}
// Gather bounded compact-table indexes and widen their 24-bit values.
U gather_chunks(const uint32_t *p, U index) {
  return U(_mm512_cvtepu32_epi64(_mm512_i64gather_epi32(__m512i(index), p, 4)));
}
// Exact unsigned division by 24 for 32-bit inputs, keeping quotient calculation in SIMD.
U divide24(U value) {
  return U(_mm512_mul_epu32(__m512i(value), _mm512_set1_epi64(0xaaaaaaabULL))) >> 36;
}
// Split eight FP64 lanes into VGPR words with AVX512 masked stores.
void publish(uint32_t *const *d, int lane, uint64_t exec_mask, U value) {
  for (int reg = 0; reg < 2; ++reg) {
    const auto words = _mm512_cvtepi64_epi32(__m512i(value >> (32 * reg)));
    _mm256_mask_storeu_epi32(d[reg] + lane, __mmask8(exec_mask), words);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
#include "math/x86_64/cdna_preop_impl.h"

namespace rocjitsu::arith {
// Evaluate and publish a complete wave using the named ISA and validated controls.
void cdna_preop_x86_64_v4(int lanes, const uint32_t *a_hi, const uint32_t *b,
                          const uint32_t *chunks, int omod, unsigned rounding, bool clamp,
                          uint64_t exec_mask, uint32_t *const *d) {
  RJ_ARITH_IMPL_NAMESPACE::evaluate(lanes, a_hi, b, chunks, omod, rounding, clamp, exec_mask, d);
}
} // namespace rocjitsu::arith
