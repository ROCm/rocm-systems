// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 vector INT16/INT8/INT4 dot products and staged integer saturation.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/compiler.h"
#include "common/simd.h"
#include "dot/operand_halves.h"
#include "integer/x86_64/packed_integer_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen pairs to signed 16-bit factors, including unsigned bytes up to 255.
// The full dot fits int32; only the final addition of C can overflow.
template <int Bits, bool Unsigned>
void dot(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  __m512i select_a{}, select_b{};
  if (instruction_flags & kDotHalfSelection) {
    select_a = _mm512_broadcast_i64x4(packed_integer_selection(
        instruction_flags & RJ_ARITH_DOT_LO_A_HIGH, instruction_flags & RJ_ARITH_DOT_HI_A_LOW));
    select_b = _mm512_broadcast_i64x4(packed_integer_selection(
        instruction_flags & RJ_ARITH_DOT_LO_B_HIGH, instruction_flags & RJ_ARITH_DOT_HI_B_LOW));
  }
  const auto sign_a = _mm512_set1_epi32(
      !Unsigned && (instruction_flags & RJ_ARITH_DOT_SIGNED_A) ? 1 << (Bits - 1) : 0);
  const auto sign_b = _mm512_set1_epi32(
      !Unsigned && (instruction_flags & RJ_ARITH_DOT_SIGNED_B) ? 1 << (Bits - 1) : 0);
  const auto fields = _mm512_set1_epi32(((1 << Bits) - 1) * 0x10001);
  const auto bias_a = _mm512_or_si512(sign_a, _mm512_slli_epi32(sign_a, 16));
  const auto bias_b = _mm512_or_si512(sign_b, _mm512_slli_epi32(sign_b, 16));
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    if (instruction_flags & kDotHalfSelection) {
      // Select halves inside the arithmetic registers, without staging a wave.
      va = _mm512_shuffle_epi8(va, select_a);
      vb = _mm512_shuffle_epi8(vb, select_b);
    }
    auto acc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
    auto sum = _mm512_setzero_si512();
    RJ_ARITH_UNROLL(4)
    for (int shift = 0; shift < 16; shift += Bits) {
      // Pair corresponding fields in the low and high halfwords.
      auto x = _mm512_and_si512(_mm512_srl_epi32(va, _mm_cvtsi32_si128(shift)), fields);
      auto y = _mm512_and_si512(_mm512_srl_epi32(vb, _mm_cvtsi32_si128(shift)), fields);
      x = _mm512_sub_epi16(_mm512_xor_si512(x, bias_a), bias_a);
      y = _mm512_sub_epi16(_mm512_xor_si512(y, bias_b), bias_b);
      sum = _mm512_add_epi32(sum, _mm512_madd_epi16(x, y));
    }
    auto result = _mm512_add_epi32(acc, sum);
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      if constexpr (Unsigned) {
        auto overflow = _mm512_cmpgt_epu32_mask(acc, result);
        result = _mm512_mask_mov_epi32(result, overflow, _mm512_set1_epi32(-1));
      } else {
        auto overflow =
            _mm512_and_si512(_mm512_xor_si512(acc, result), _mm512_xor_si512(sum, result));
        auto limit = _mm512_xor_si512(_mm512_srai_epi32(acc, 31), _mm512_set1_epi32(INT32_MAX));
        result = _mm512_mask_mov_epi32(result, _mm512_movepi32_mask(overflow), limit);
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

// Accumulate two full-width products before saturation. Signed DOT2 can exceed
// INT32_MAX before adding C, so clamping needs 64-bit lanes even for signed factors.
template <bool Unsigned>
void dot16(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
           const uint32_t *b, const uint32_t *c) {
  __m512i select_a{}, select_b{};
  if (instruction_flags & kDotHalfSelection) {
    select_a = _mm512_broadcast_i64x4(packed_integer_selection(
        instruction_flags & RJ_ARITH_DOT_LO_A_HIGH, instruction_flags & RJ_ARITH_DOT_HI_A_LOW));
    select_b = _mm512_broadcast_i64x4(packed_integer_selection(
        instruction_flags & RJ_ARITH_DOT_LO_B_HIGH, instruction_flags & RJ_ARITH_DOT_HI_B_LOW));
  }
  // Without saturation only the low 32 bits matter, so full-width vector multiplies
  // avoid the widening needed below without changing signed or unsigned wrapping.
  if (!(instruction_flags & RJ_ARITH_DOT_CLAMP)) {
    for (int lane = 0; lane < 32; lane += 16) {
      auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
      auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
      if (instruction_flags & kDotHalfSelection) {
        // Select halves inside the arithmetic registers, without staging a wave.
        va = _mm512_shuffle_epi8(va, select_a);
        vb = _mm512_shuffle_epi8(vb, select_b);
      }
      auto vc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
      __m512i sum;
      if constexpr (Unsigned) {
        auto low_a = _mm512_and_si512(va, _mm512_set1_epi32(0xffff));
        auto low_b = _mm512_and_si512(vb, _mm512_set1_epi32(0xffff));
        sum = _mm512_add_epi32(
            _mm512_mullo_epi32(low_a, low_b),
            _mm512_mullo_epi32(_mm512_srli_epi32(va, 16), _mm512_srli_epi32(vb, 16)));
      } else {
        // PMADDWD's sole overflow case (two INT16_MIN products) wraps to
        // 0x80000000, exactly as required when CLAMP is clear.
        sum = _mm512_madd_epi16(va, vb);
      }
      auto result = _mm512_add_epi32(vc, sum);
      _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
    }
    return;
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    if (instruction_flags & kDotHalfSelection) {
      // Select halves inside the arithmetic registers, without staging a wave.
      va = _mm256_shuffle_epi8(va, _mm512_castsi512_si256(select_a));
      vb = _mm256_shuffle_epi8(vb, _mm512_castsi512_si256(select_b));
    }
    auto vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto low = [](__m256i v) {
      if constexpr (Unsigned) {
        return _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
      } else {
        return _mm256_srai_epi32(_mm256_slli_epi32(v, 16), 16);
      }
    };
    auto high = [](__m256i v) {
      if constexpr (Unsigned) {
        return _mm256_srli_epi32(v, 16);
      } else {
        return _mm256_srai_epi32(v, 16);
      }
    };
    auto widen = [](__m256i v) {
      if constexpr (Unsigned) {
        return _mm512_cvtepu32_epi64(v);
      } else {
        return _mm512_cvtepi32_epi64(v);
      }
    };
    auto product = [](__m512i x, __m512i y) {
      if constexpr (Unsigned) {
        return _mm512_mul_epu32(x, y);
      } else {
        return _mm512_mul_epi32(x, y);
      }
    };
    auto sum =
        _mm512_add_epi64(widen(vc), _mm512_add_epi64(product(widen(low(va)), widen(low(vb))),
                                                     product(widen(high(va)), widen(high(vb)))));
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      const auto upper = _mm512_set1_epi64(Unsigned ? int64_t(UINT32_MAX) : INT32_MAX);
      sum = _mm512_min_epi64(sum, upper);
      if constexpr (!Unsigned) {
        const auto lower = _mm512_set1_epi64(INT32_MIN);
        sum = _mm512_max_epi64(sum, lower);
      }
    }
    auto result = _mm512_cvtepi64_epi32(sum);
    _mm256_mask_storeu_epi32(d + lane, __mmask8(exec_mask >> lane), result);
  }
}

} // namespace

// Accumulate packed integer dot products. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void integer_dot_x86_64_v4(int bits, bool unsigned_acc, uint32_t exec_mask,
                           uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c) {
  if (bits == 16) {
    if (unsigned_acc) {
      dot16<true>(exec_mask, instruction_flags, d, a, b, c);
    } else {
      dot16<false>(exec_mask, instruction_flags, d, a, b, c);
    }
  } else if (bits == 8) {
    if (unsigned_acc) {
      dot<8, true>(exec_mask, instruction_flags, d, a, b, c);
    } else {
      dot<8, false>(exec_mask, instruction_flags, d, a, b, c);
    }
  } else {
    if (unsigned_acc) {
      dot<4, true>(exec_mask, instruction_flags, d, a, b, c);
    } else {
      dot<4, false>(exec_mask, instruction_flags, d, a, b, c);
    }
  }
}

} // namespace rocjitsu::arith
