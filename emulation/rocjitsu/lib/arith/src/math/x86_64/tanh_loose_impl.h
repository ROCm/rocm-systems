// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Shared Loose TANH approximation. Small arguments avoid exponential cancellation.
#pragma once
#include "common/isa.h"
#include <immintrin.h>
#include <stdint.h>
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
using F8 [[gnu::vector_size(32)]] = float;
using I8 [[gnu::vector_size(32)]] = int32_t;
// Round reduced exponents locally; no host FP-control changes.
inline F8 nearest(F8 x) {
  return (F8)_mm256_round_ps((__m256)x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
inline I8 integers(F8 x) { return (I8)_mm256_cvttps_epi32((__m256)x); }
#if defined(__AVX512F__)
using F16 [[gnu::vector_size(64)]] = float;
using I16 [[gnu::vector_size(64)]] = int32_t;
inline F16 nearest(F16 x) {
  return (F16)_mm512_roundscale_ps((__m512)x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
inline I16 integers(F16 x) { return (I16)_mm512_cvttps_epi32((__m512)x); }
#endif

// Approximate tanh on finite magnitudes, restoring signs and quiet NaN payloads at the end.
template <typename V, typename I> inline I tanh_vector(I raw) {
  const I magnitude = raw & 0x7fffffff;
  const V zero = V{};
  const V one = zero + 1.0f;
  // Large magnitudes saturate; sanitize NaNs before floating arithmetic and conversions.
  const V x = magnitude > 0x41100000 ? zero + 9.0f : (V)magnitude;
  const V exponent = x * -2.8853900817779268f;
  const V whole = nearest(exponent), fraction = exponent - whole;
  // Same reduced exp2 Taylor coefficients as the existing Loose exponential path.
  V p = zero + 1.525273380405984e-5f;
  p = p * fraction + 0.0001540353039338161f;
  p = p * fraction + 0.001333355814642844f;
  p = p * fraction + 0.009618129107628477f;
  p = p * fraction + 0.05550410866482158f;
  p = p * fraction + 0.2402265069591007f;
  p = p * fraction + 0.6931471805599453f;
  p = p * fraction + 1.0f;
  const V exponential = p * (V)((integers(whole) + 127) << 23);
  V value = (one - exponential) / (one + exponential);
  const V small = x < 0.25f ? x : zero;
  const V square = small * small;
  // Odd Taylor polynomial through x^9; the next term at 1/4 is below 3e-9.
  const V correction =
      ((square * (62.0f / 2835.0f) - 17.0f / 315.0f) * square + 2.0f / 15.0f) * square -
      1.0f / 3.0f;
  value = x < 0.25f ? small + small * square * correction : value;
  I bits = (I)value | (raw & int32_t(0x80000000U));
  return magnitude > 0x7f800000 ? raw | 0x400000 : bits;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
