// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_kernels.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_kernels_detail.h"

#include <immintrin.h>

namespace rocjitsu::amdgpu::fp_math {
namespace {

struct Avx512Ops {
  using V = __m512i;
  using M = __mmask8;

  static V set(std::uint64_t value) { return _mm512_set1_epi64(static_cast<long long>(value)); }
  static V add(V a, V b) { return _mm512_add_epi64(a, b); }
  static V sub(V a, V b) { return _mm512_sub_epi64(a, b); }
  static V mul(V a, V b) { return _mm512_mullo_epi64(a, b); }
  static V bit_or(V a, V b) { return _mm512_or_si512(a, b); }
  static V bit_xor(V a, V b) { return _mm512_xor_si512(a, b); }
  static V band(V a, V b) { return _mm512_and_si512(a, b); }
  static V shr(V a, unsigned amount) {
    return _mm512_srl_epi64(a, _mm_cvtsi32_si128(static_cast<int>(amount)));
  }
  static V shl(V a, unsigned amount) {
    return _mm512_sll_epi64(a, _mm_cvtsi32_si128(static_cast<int>(amount)));
  }
  static V sar(V a, unsigned amount) {
    return _mm512_sra_epi64(a, _mm_cvtsi32_si128(static_cast<int>(amount)));
  }
  static V shrv(V a, V amount) { return _mm512_srlv_epi64(a, amount); }
  static V shlv(V a, V amount) { return _mm512_sllv_epi64(a, amount); }
  static M eq(V a, V b) { return _mm512_cmpeq_epi64_mask(a, b); }
  static M ne(V a, V b) { return _mm512_cmpneq_epi64_mask(a, b); }
  static M lt_s(V a, V b) { return _mm512_cmplt_epi64_mask(a, b); }
  static M ge_s(V a, V b) { return _mm512_cmpge_epi64_mask(a, b); }
  static M lt_u(V a, V b) { return _mm512_cmplt_epu64_mask(a, b); }
  static M gt_u(V a, V b) { return _mm512_cmpgt_epu64_mask(a, b); }
  static M ge_u(V a, V b) { return _mm512_cmpge_epu64_mask(a, b); }
  static M le_u(V a, V b) { return _mm512_cmple_epu64_mask(a, b); }
  static M mask_and(M a, M b) { return a & b; }
  static M mask_or(M a, M b) { return a | b; }
  static M mask_not(M a) { return static_cast<M>(~a); }
  static bool mask_any(M mask) { return mask != 0; }
  static V select(M mask, V yes, V no) { return _mm512_mask_blend_epi64(mask, no, yes); }
  static V min_u(V a, V b) { return _mm512_min_epu64(a, b); }
  static V lzcnt(V a) { return _mm512_lzcnt_epi64(a); }

  static V load(const std::uint32_t *input) {
    return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(input)));
  }
  static void store(std::uint32_t *output, V result) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), _mm512_cvtepi64_epi32(result));
  }

  template <bool Logarithm, unsigned Field> static V coefficient(V index) {
    const std::uint32_t *table;
    if constexpr (Logarithm)
      table = detail::log_columns[Field].data();
    else
      table = detail::exp_columns[Field].data();
    const V indices = _mm512_zextsi256_si512(_mm512_cvtepi64_epi32(index));
    const V values = _mm512_permutex2var_epi32(_mm512_loadu_si512(table), indices,
                                               _mm512_loadu_si512(table + 16));
    V result = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(values));
    if constexpr (Logarithm)
      result = select(eq(index, set(32)), set(table[32]), result);
    return result;
  }
};

template <bool Logarithm> F32Bits8 run8(F32Bits8 input, bool quiet_snan) {
  F32Bits8 output{};
  Avx512Ops::store(output.lane.data(), detail::ExpLog<Avx512Ops>::template batch<Logarithm>(
                                           Avx512Ops::load(input.lane.data()), quiet_snan));
  return output;
}

template <bool Logarithm> F32Bits16 run16(F32Bits16 input, bool quiet_snan) {
  // Two independent eight-lane AVX-512 integer packs. This is only an x16
  // call boundary, not a claim of sixteen simultaneous 64-bit SIMD lanes.
  F32Bits16 output{};
  for (unsigned base = 0; base < 16; base += 8) {
    Avx512Ops::store(output.lane.data() + base,
                     detail::ExpLog<Avx512Ops>::template batch<Logarithm>(
                         Avx512Ops::load(input.lane.data() + base), quiet_snan));
  }
  return output;
}

} // namespace

F32Bits8 exp_v4x8(F32Bits8 input, bool quiet_snan) noexcept {
  return run8<false>(input, quiet_snan);
}
F32Bits8 log_v4x8(F32Bits8 input, bool quiet_snan) noexcept {
  return run8<true>(input, quiet_snan);
}
F32Bits16 exp_v4x16(F32Bits16 input, bool quiet_snan) noexcept {
  return run16<false>(input, quiet_snan);
}
F32Bits16 log_v4x16(F32Bits16 input, bool quiet_snan) noexcept {
  return run16<true>(input, quiet_snan);
}

} // namespace rocjitsu::amdgpu::fp_math
