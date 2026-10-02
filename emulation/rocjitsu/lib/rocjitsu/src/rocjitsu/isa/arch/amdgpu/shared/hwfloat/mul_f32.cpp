// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32.h"

#include <bit>
#include <cassert>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace rocjitsu::amdgpu::hwfloat {

namespace {

uint64_t lanes_mask(uint32_t lanes) {
  return lanes >= 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
}

uint32_t multiply_lanes(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result, uint64_t lanes,
                        MulF32Policy policy) {
  uint32_t causes = 0;
  while (lanes) {
    const int lane = std::countr_zero(lanes);
    lanes &= lanes - 1;
    const MulF32Result product = multiply_f32(lhs[lane], rhs[lane], policy);
    result[lane] = product.bits;
    causes |= plain_mul_f32_causes(product, policy);
  }
  return causes;
}

#if defined(__x86_64__)

// Callers must check mul_f32_avx512_available(): the compiler may schedule
// AVX-512 instructions anywhere in this function, including ahead of a check
// placed inside it.
[[gnu::target("avx512f"), gnu::noinline]] uint32_t
multiply_avx512(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result, uint64_t active,
                uint32_t lanes, MulF32Policy policy) {
  const __m512i exponent_mask = _mm512_set1_epi32(255);
  const __m512i one = _mm512_set1_epi32(0x3f800000);
  const __m512i fraction_mask = _mm512_set1_epi32(0x007fffff);
  const __m512i hidden_bit = _mm512_set1_epi32(0x00800000);
  const __m512i discarded = _mm512_set1_epi64(0x7fffff);
  const __m512i extra_discarded = _mm512_set1_epi64(int64_t{1} << 23);
  const __m512i high_bit = _mm512_set1_epi64(int64_t{1} << 47);
  const __m512i zero = _mm512_setzero_si512();
  uint32_t causes = 0;
  for (uint32_t base = 0; base < lanes; base += 16) {
    const auto block_active = static_cast<__mmask16>(active >> base);
    if (!block_active)
      continue;
    const auto loaded = static_cast<__mmask16>(lanes_mask(lanes - base));
    const __m512i a = _mm512_maskz_loadu_epi32(loaded, lhs + base);
    const __m512i b = _mm512_maskz_loadu_epi32(loaded, rhs + base);
    const __m512i ae = _mm512_and_si512(_mm512_srli_epi32(a, 23), exponent_mask);
    const __m512i be = _mm512_and_si512(_mm512_srli_epi32(b, 23), exponent_mask);
    const __m512i sum = _mm512_add_epi32(ae, be);
    // Normal sources with this exponent sum have a normal, finite product, so
    // host denormal controls cannot apply and tininess cannot occur.
    const __mmask16 safe = block_active & _mm512_cmp_epu32_mask(ae, zero, _MM_CMPINT_GT) &
                           _mm512_cmp_epu32_mask(be, zero, _MM_CMPINT_GT) &
                           _mm512_cmp_epu32_mask(ae, exponent_mask, _MM_CMPINT_LT) &
                           _mm512_cmp_epu32_mask(be, exponent_mask, _MM_CMPINT_LT) &
                           _mm512_cmp_epu32_mask(sum, _mm512_set1_epi32(128), _MM_CMPINT_GE) &
                           _mm512_cmp_epu32_mask(sum, _mm512_set1_epi32(380), _MM_CMPINT_LE);
    const __m512 left = _mm512_castsi512_ps(_mm512_mask_mov_epi32(one, safe, a));
    const __m512 right = _mm512_castsi512_ps(_mm512_mask_mov_epi32(one, safe, b));
    __m512 product;
    switch (policy.round) {
    case 1:
      product = _mm512_mul_round_ps(left, right, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
      break;
    case 2:
      product = _mm512_mul_round_ps(left, right, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
      break;
    case 3:
      product = _mm512_mul_round_ps(left, right, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      break;
    default:
      product = _mm512_mul_round_ps(left, right, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      break;
    }
    _mm512_mask_storeu_epi32(result + base, safe, _mm512_castps_si512(product));

    // Exact 24x24-bit significand products of even and odd lanes, with unsafe
    // lanes zeroed. A product with bit 47 set discards 24 bits, otherwise 23.
    const __m512i as = _mm512_maskz_mov_epi32(
        safe, _mm512_or_si512(_mm512_and_si512(a, fraction_mask), hidden_bit));
    const __m512i bs = _mm512_maskz_mov_epi32(
        safe, _mm512_or_si512(_mm512_and_si512(b, fraction_mask), hidden_bit));
    const __m512i even = _mm512_mul_epu32(as, bs);
    const __m512i odd = _mm512_mul_epu32(_mm512_srli_epi64(as, 32), _mm512_srli_epi64(bs, 32));
    const __mmask8 inexact =
        _mm512_test_epi64_mask(even, discarded) |
        (_mm512_test_epi64_mask(even, high_bit) & _mm512_test_epi64_mask(even, extra_discarded)) |
        _mm512_test_epi64_mask(odd, discarded) |
        (_mm512_test_epi64_mask(odd, high_bit) & _mm512_test_epi64_mask(odd, extra_discarded));
    if (inexact)
      causes |= mul_f32_cause::kInexact;

    const auto slow = static_cast<uint16_t>(block_active & ~safe);
    causes |= multiply_lanes(lhs + base, rhs + base, result + base, slow, policy);
  }
  return causes;
}

#endif

} // namespace

uint32_t multiply_f32_wave_integer(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                                   uint64_t active, uint32_t lanes, MulF32Policy policy) {
  assert(lanes <= kMulF32MaxLanes);
  return multiply_lanes(lhs, rhs, result, active & lanes_mask(lanes), policy);
}

bool mul_f32_avx512_available() {
#if defined(__x86_64__)
  // A load from the compiler runtime's feature table, initialized before
  // static constructors; it reports AVX512F only when XCR0 enables the opmask
  // and ZMM state.
  return __builtin_cpu_supports("avx512f");
#else
  return false;
#endif
}

uint32_t multiply_f32_wave_avx512(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                                  uint64_t active, uint32_t lanes, MulF32Policy policy) {
  assert(lanes <= kMulF32MaxLanes);
#if defined(__x86_64__)
  if (mul_f32_avx512_available())
    return multiply_avx512(lhs, rhs, result, active & lanes_mask(lanes), lanes, policy);
#endif
  return multiply_f32_wave_integer(lhs, rhs, result, active, lanes, policy);
}

uint32_t multiply_f32_wave(const uint32_t *lhs, const uint32_t *rhs, uint32_t *result,
                           uint64_t active, uint32_t lanes, MulF32Policy policy) {
  return multiply_f32_wave_avx512(lhs, rhs, result, active, lanes, policy);
}

} // namespace rocjitsu::amdgpu::hwfloat
