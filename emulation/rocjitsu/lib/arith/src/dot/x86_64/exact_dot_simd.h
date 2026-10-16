// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD vector integer primitives used by empirical exact DOT and WMMA models.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot {

// Width-specific integer operations for the shared exact DOT/WMMA model.
// Comparisons return all-bit lane masks at either width. Variable shifts retain
// x86's zero result for counts >=32, which the alignment algorithm requires.
template <typename V> inline V constant(int value);
template <typename V> inline V load(const void *p);

template <> inline __m256i constant<__m256i>(int value) { return _mm256_set1_epi32(value); }

template <> inline __m256i load<__m256i>(const void *p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
}

// Write vector values to destination storage.
inline void store(void *p, __m256i value) {
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), value);
}

// Add corresponding integer lanes with unsigned wraparound.
inline __m256i add(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }

// Subtract corresponding integer lanes with unsigned wraparound.
inline __m256i sub(__m256i a, __m256i b) { return _mm256_sub_epi32(a, b); }

// Multiply corresponding integer lanes, retaining the low result bits.
inline __m256i mul(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }

// Select the greater signed integer value in each lane.
inline __m256i maximum(__m256i a, __m256i b) { return _mm256_max_epi32(a, b); }

// Return the bitwise intersection of the vector operands.
inline __m256i bit_and(__m256i a, __m256i b) { return _mm256_and_si256(a, b); }

// Return the bitwise union of the vector operands.
inline __m256i bit_or(__m256i a, __m256i b) { return _mm256_or_si256(a, b); }

// Return the bitwise exclusive-or of the vector operands.
inline __m256i bit_xor(__m256i a, __m256i b) { return _mm256_xor_si256(a, b); }

// Return bits from b whose corresponding bits in a are clear.
inline __m256i andnot(__m256i a, __m256i b) { return _mm256_andnot_si256(a, b); }

// Shift integer lanes left, discarding bits outside the lane width.
inline __m256i shift_left(__m256i a, __m256i b) { return _mm256_sllv_epi32(a, b); }

// Logically shift integer lanes right, inserting zeros.
inline __m256i shift_right(__m256i a, __m256i b) { return _mm256_srlv_epi32(a, b); }

// Shift integer lanes left, discarding bits outside the lane width.
inline __m256i shift_left(__m256i a, int count) { return _mm256_slli_epi32(a, count); }

// Logically shift integer lanes right, inserting zeros.
inline __m256i shift_right(__m256i a, int count) { return _mm256_srli_epi32(a, count); }

// Arithmetically shift signed lanes right, extending their signs.
inline __m256i signed_shift_right(__m256i a, int count) { return _mm256_srai_epi32(a, count); }

// Return signed integer magnitudes lane by lane.
inline __m256i absolute(__m256i a) { return _mm256_abs_epi32(a); }

// Return the lane mask for equal operands.
inline __m256i equal(__m256i a, __m256i b) { return _mm256_cmpeq_epi32(a, b); }

// Return the lane mask for a > b.
inline __m256i greater(__m256i a, __m256i b) { return _mm256_cmpgt_epi32(a, b); }

// Convert signed integer lanes to FP32 and return their biased exponent fields.
inline __m256i float_exponent(__m256i a) {
  return _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(a)), 23);
}

// Conversion may round up to a power of two. Correct that one-bit exponent
// error explicitly; DOT magnitudes fit signed i32. Zero's exponent is
// unspecified and discarded by packing.
inline __m256i highest_bit(__m256i magnitude) {
  auto top = sub(float_exponent(magnitude), constant<__m256i>(127));
  auto carry = greater(shift_left(constant<__m256i>(1), top), magnitude);
  return add(top, carry);
}

// Collect vector predicate sign bits into a scalar lane mask.
inline int lane_mask(__m256i value) { return _mm256_movemask_ps(_mm256_castsi256_ps(value)); }

// Select values lane by lane according to the supplied predicate.
inline __m256i choose(__m256i mask, __m256i yes, __m256i no) {
  return _mm256_blendv_epi8(no, yes, mask);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::exact_dot
