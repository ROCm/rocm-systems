// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD vector empirical reciprocal evaluation, corrections, and exceptional-lane repair.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "math/rcp_model.h"
#include "math/x86_64/sfu_exact_x86_64.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::rcp_simd {

using sfu_simd::U4;
#if defined(__AVX512F__)
using sfu_simd::U8;
#endif
using sfu_simd::choose;
using sfu_simd::load;
using sfu_simd::output;
using sfu_simd::store;

// Select the requested polynomial coefficient field for each reduced input.
template <class V> V coefficient(V mantissa, int field) {
  return sfu_simd::lookup32(mantissa >> 18, detail::rcp_coefficients[field]);
}

// Every inherited bucket fits in one AVX512 load or two AVX2 loads.
static_assert([] {
  for (int bucket = 0; bucket < 256; ++bucket) {
    int count = detail::kAmdgpuRcpCorrectionOffsets[bucket + 1] -
                detail::kAmdgpuRcpCorrectionOffsets[bucket];
    if (count > 32) {
      return false;
    }
  }
  return true;
}());

// Reassemble the inherited reciprocal result and apply unary output modifiers.
// Reciprocal flushes input and output subnormals regardless of guest MODE.
template <class V> V finish(V input, V normalized, uint64_t flags, uint32_t mode, bool quiet_snan) {
  V sign = input & 0x80000000ULL, magnitude = input & 0x7fffffffULL;
  V exponent = (normalized >> 23) + 127ULL - (magnitude >> 23);
  V value = sign | (exponent << 23) | (normalized & 0x7fffffULL);

  // Exponent subtraction wraps in unsigned lanes; its high bit marks underflow.
  V negative_exponent = (V)(exponent > 0x7fffffffffffffffULL);
  value = choose((V)(exponent >= 255ULL), sign | 0x7f800000ULL, value);
  value = choose(negative_exponent | (V)(exponent == 0ULL), sign, value);
  value = choose((V)(magnitude < 0x800000ULL), sign | 0x7f800000ULL, value);
  value = choose((V)(magnitude == 0x7f800000ULL), sign, value);
  value =
      choose((V)(magnitude > 0x7f800000ULL), input | uint64_t(quiet_snan ? 0x400000U : 0U), value);

  return output(value, flags, mode);
}

// Evaluate the reciprocal polynomial for normalized mantissas.
template <class V> V polynomial(V mantissa) {
  return sfu_simd::reciprocal_polynomial<false>(mantissa, coefficient(mantissa, 0),
                                                coefficient(mantissa, 1), coefficient(mantissa, 2));
}

// Compute empirical exact reciprocals with modifiers and guest denormal rules; preserve
// inactive D lanes.
template <class V>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         bool quiet_snan) {
  constexpr int Lanes = sizeof(V) / sizeof(uint64_t);
  uint32_t result[32];
  for (int base = 0; base < 32; base += Lanes) {
    V input = load(a + base, V{});
    if (mode & RJ_ARITH_ALU_ABS_A) {
      input &= 0x7fffffffULL;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      input ^= 0x80000000ULL;
    }
    V mantissas = input & 0x7fffffULL;
    V values =
        sfu_simd::correct(mantissas, polynomial(mantissas), detail::kAmdgpuRcpCorrectionOffsets,
                          detail::kAmdgpuRcpCorrections);
    store(result + base, finish(input, values, flags, mode, quiet_snan));
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
}

} // namespace rocjitsu::arith::rcp_simd
