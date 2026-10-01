// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_kernels.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_kernels_detail.h"

#include <immintrin.h>

namespace rocjitsu::amdgpu::fp_math {
namespace {

// AVX2 has four 64-bit integer lanes. Each x8 invocation processes two
// four-lane packs; the integer approximation remains packed throughout.
struct Avx2Ops {
  using V = __m256i;
  using M = __m256i;

  static V set(std::uint64_t value) { return _mm256_set1_epi64x(static_cast<long long>(value)); }
  static V add(V a, V b) { return _mm256_add_epi64(a, b); }
  static V sub(V a, V b) { return _mm256_sub_epi64(a, b); }
  static V bit_or(V a, V b) { return _mm256_or_si256(a, b); }
  static V bit_xor(V a, V b) { return _mm256_xor_si256(a, b); }
  static V band(V a, V b) { return _mm256_and_si256(a, b); }

  // AVX2 has 32x32->64 multiplication, but no packed 64x64 low product.
  // The cross terms recover the low 64 bits modulo 2^64 in every lane.
  static V mul(V a, V b) {
    const V low = _mm256_mul_epu32(a, b);
    const V cross = add(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), b),
                        _mm256_mul_epu32(a, _mm256_srli_epi64(b, 32)));
    return add(low, _mm256_slli_epi64(cross, 32));
  }

  static V shr(V a, unsigned amount) {
    return _mm256_srl_epi64(a, _mm_cvtsi32_si128(static_cast<int>(amount)));
  }
  static V shl(V a, unsigned amount) {
    return _mm256_sll_epi64(a, _mm_cvtsi32_si128(static_cast<int>(amount)));
  }
  static V sar(V a, unsigned amount) {
    if (amount == 0)
      return a;
    const V sign = _mm256_cmpgt_epi64(set(0), a);
    return bit_or(shr(a, amount), shl(sign, 64 - amount));
  }
  static V shrv(V a, V amount) { return _mm256_srlv_epi64(a, amount); }
  static V shlv(V a, V amount) { return _mm256_sllv_epi64(a, amount); }

  static M eq(V a, V b) { return _mm256_cmpeq_epi64(a, b); }
  static M ne(V a, V b) { return mask_not(eq(a, b)); }
  static M lt_s(V a, V b) { return _mm256_cmpgt_epi64(b, a); }
  static M ge_s(V a, V b) { return mask_not(lt_s(a, b)); }
  static M lt_u(V a, V b) {
    const V flip = set(std::uint64_t{1} << 63);
    return lt_s(bit_xor(a, flip), bit_xor(b, flip));
  }
  static M gt_u(V a, V b) { return lt_u(b, a); }
  static M ge_u(V a, V b) { return mask_not(lt_u(a, b)); }
  static M le_u(V a, V b) { return mask_not(gt_u(a, b)); }
  static M mask_and(M a, M b) { return _mm256_and_si256(a, b); }
  static M mask_or(M a, M b) { return _mm256_or_si256(a, b); }
  static M mask_not(M a) { return _mm256_xor_si256(a, set(~std::uint64_t{0})); }
  static bool mask_any(M mask) { return _mm256_movemask_pd(_mm256_castsi256_pd(mask)) != 0; }
  static V select(M mask, V yes, V no) { return _mm256_blendv_epi8(no, yes, mask); }
  static V min_u(V a, V b) { return select(lt_u(a, b), a, b); }

  static V lzcnt(V a) {
    V x = a;
    V count = set(0);
    constexpr unsigned steps[] = {32u, 16u, 8u, 4u, 2u, 1u};
    for (unsigned step : steps) {
      const M zero_high = eq(shr(x, 64 - step), set(0));
      count = add(count, select(zero_high, set(step), set(0)));
      x = select(zero_high, shl(x, step), x);
    }
    return add(count, select(eq(x, set(0)), set(1), set(0)));
  }

  static __m128i pack_low_dwords(V value) {
    const V interleaved = _mm256_shuffle_epi32(value, _MM_SHUFFLE(2, 0, 2, 0));
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(interleaved),
                              _mm256_extracti128_si256(interleaved, 1));
  }
  static V load(const std::uint32_t *input) {
    return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(input)));
  }
  static void store(std::uint32_t *output, V result) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output), pack_low_dwords(result));
  }

  template <bool Logarithm, unsigned Field> static V coefficient(V index) {
    const std::uint32_t *column;
    if constexpr (Logarithm)
      column = detail::log_columns[Field].data();
    else
      column = detail::exp_columns[Field].data();
    const __m128i gathered = _mm_i32gather_epi32(reinterpret_cast<const int *>(column),
                                                 pack_low_dwords(index), sizeof(std::uint32_t));
    return _mm256_cvtepu32_epi64(gathered);
  }
};

template <bool Logarithm> F32Bits8 run(F32Bits8 input, bool quiet_snan) {
  F32Bits8 output{};
  for (unsigned base = 0; base < 8; base += 4)
    Avx2Ops::store(output.lane.data() + base,
                   detail::ExpLog<Avx2Ops>::template batch<Logarithm>(
                       Avx2Ops::load(input.lane.data() + base), quiet_snan));
  return output;
}

} // namespace

F32Bits8 exp_v3(F32Bits8 input, bool quiet_snan) noexcept { return run<false>(input, quiet_snan); }
F32Bits8 log_v3(F32Bits8 input, bool quiet_snan) noexcept { return run<true>(input, quiet_snan); }

} // namespace rocjitsu::amdgpu::fp_math
