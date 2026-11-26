// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 compact CDNA range-reduction lookup and integer FP64 encoding.
#include "common/isa.h"
#include "math/cdna_trig_preop.h"
#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
using U = uint64_t __attribute__((vector_size(32)));
// Widen unsigned VGPR words to arithmetic lanes.
U load_words(const uint32_t *p) {
  return U(_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))));
}
// Gather bounded compact-table indexes and widen their 24-bit values.
U gather_chunks(const uint32_t *p, U index) {
  return U(_mm256_cvtepu32_epi64(
      _mm256_i64gather_epi32(reinterpret_cast<const int *>(p), __m256i(index), 4)));
}
// Exact unsigned division by 24 for 32-bit inputs, keeping quotient calculation in SIMD.
U divide24(U value) {
  return U(_mm256_mul_epu32(__m256i(value), _mm256_set1_epi64x(0xaaaaaaabULL))) >> 36;
}
// Split four FP64 lanes into VGPR words and preserve inactive lanes with masked stores.
void publish(uint32_t *const *d, int lane, uint64_t exec_mask, U value) {
  const auto mask = _mm_sllv_epi32(_mm_set1_epi32(int(exec_mask)), _mm_setr_epi32(31, 30, 29, 28));
  for (int reg = 0; reg < 2; ++reg) {
    const auto words = _mm256_shuffle_epi32(__m256i(value >> (32 * reg)), _MM_SHUFFLE(2, 0, 2, 0));
    const auto packed =
        _mm_unpacklo_epi64(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
    _mm_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), mask, packed);
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
#include "math/x86_64/cdna_preop_impl.h"

namespace rocjitsu::arith {
// Evaluate and publish a complete wave using the named ISA and validated controls.
void cdna_preop_x86_64_v3(int lanes, const uint32_t *a_hi, const uint32_t *b,
                          const uint32_t *chunks, int omod, unsigned rounding, bool clamp,
                          uint64_t exec_mask, uint32_t *const *d) {
  RJ_ARITH_IMPL_NAMESPACE::evaluate(lanes, a_hi, b, chunks, omod, rounding, clamp, exec_mask, d);
}
} // namespace rocjitsu::arith
