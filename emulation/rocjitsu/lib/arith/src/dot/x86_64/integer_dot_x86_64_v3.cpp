// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 vector INT16/INT8/INT4 dot products and staged integer saturation.
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
  __m256i select_a{}, select_b{};
  if (instruction_flags & kDotHalfSelection) {
    select_a = packed_integer_selection(instruction_flags & RJ_ARITH_DOT_LO_A_HIGH,
                                        instruction_flags & RJ_ARITH_DOT_HI_A_LOW);
    select_b = packed_integer_selection(instruction_flags & RJ_ARITH_DOT_LO_B_HIGH,
                                        instruction_flags & RJ_ARITH_DOT_HI_B_LOW);
  }
  const auto sign_a = _mm256_set1_epi32(
      !Unsigned && (instruction_flags & RJ_ARITH_DOT_SIGNED_A) ? 1 << (Bits - 1) : 0);
  const auto sign_b = _mm256_set1_epi32(
      !Unsigned && (instruction_flags & RJ_ARITH_DOT_SIGNED_B) ? 1 << (Bits - 1) : 0);
  const auto fields = _mm256_set1_epi32(((1 << Bits) - 1) * 0x10001);
  const auto bias_a = _mm256_or_si256(sign_a, _mm256_slli_epi32(sign_a, 16));
  const auto bias_b = _mm256_or_si256(sign_b, _mm256_slli_epi32(sign_b, 16));
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    if (instruction_flags & kDotHalfSelection) {
      // Select halves inside the arithmetic registers, without staging a wave.
      va = _mm256_shuffle_epi8(va, select_a);
      vb = _mm256_shuffle_epi8(vb, select_b);
    }
    auto acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto sum = _mm256_setzero_si256();
    RJ_ARITH_UNROLL(4)
    for (int shift = 0; shift < 16; shift += Bits) {
      // Pair corresponding fields in the low and high halfwords.
      auto x = _mm256_and_si256(_mm256_srl_epi32(va, _mm_cvtsi32_si128(shift)), fields);
      auto y = _mm256_and_si256(_mm256_srl_epi32(vb, _mm_cvtsi32_si128(shift)), fields);
      x = _mm256_sub_epi16(_mm256_xor_si256(x, bias_a), bias_a);
      y = _mm256_sub_epi16(_mm256_xor_si256(y, bias_b), bias_b);
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, y));
    }
    auto result = _mm256_add_epi32(acc, sum);
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      if constexpr (Unsigned) {
        const auto sign = _mm256_set1_epi32(INT32_MIN);
        auto overflow =
            _mm256_cmpgt_epi32(_mm256_xor_si256(acc, sign), _mm256_xor_si256(result, sign));
        result = _mm256_or_si256(result, overflow);
      } else {
        auto overflow =
            _mm256_and_si256(_mm256_xor_si256(acc, result), _mm256_xor_si256(sum, result));
        auto limit = _mm256_xor_si256(_mm256_srai_epi32(acc, 31), _mm256_set1_epi32(INT32_MAX));
        result = _mm256_blendv_epi8(result, limit, _mm256_srai_epi32(overflow, 31));
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

// Accumulate two full-width products before saturation. Signed DOT2 can exceed
// INT32_MAX before adding C, so clamping needs 64-bit lanes even for signed factors.
template <bool Unsigned>
void dot16(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
           const uint32_t *b, const uint32_t *c) {
  __m256i select_a{}, select_b{};
  if (instruction_flags & kDotHalfSelection) {
    select_a = packed_integer_selection(instruction_flags & RJ_ARITH_DOT_LO_A_HIGH,
                                        instruction_flags & RJ_ARITH_DOT_HI_A_LOW);
    select_b = packed_integer_selection(instruction_flags & RJ_ARITH_DOT_LO_B_HIGH,
                                        instruction_flags & RJ_ARITH_DOT_HI_B_LOW);
  }
  // Without saturation only the low 32 bits matter, so full-width vector multiplies
  // avoid the widening needed below without changing signed or unsigned wrapping.
  if (!(instruction_flags & RJ_ARITH_DOT_CLAMP)) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
      auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      if (instruction_flags & kDotHalfSelection) {
        // Select halves inside the arithmetic registers, without staging a wave.
        va = _mm256_shuffle_epi8(va, select_a);
        vb = _mm256_shuffle_epi8(vb, select_b);
      }
      auto vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
      __m256i sum;
      if constexpr (Unsigned) {
        auto low_a = _mm256_and_si256(va, _mm256_set1_epi32(0xffff));
        auto low_b = _mm256_and_si256(vb, _mm256_set1_epi32(0xffff));
        sum = _mm256_add_epi32(
            _mm256_mullo_epi32(low_a, low_b),
            _mm256_mullo_epi32(_mm256_srli_epi32(va, 16), _mm256_srli_epi32(vb, 16)));
      } else {
        // PMADDWD's sole overflow case (two INT16_MIN products) wraps to
        // 0x80000000, exactly as required when CLAMP is clear.
        sum = _mm256_madd_epi16(va, vb);
      }
      auto result = _mm256_add_epi32(vc, sum);
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
    }
    return;
  }
  for (int lane = 0; lane < 32; lane += 4) {
    auto va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + lane));
    auto vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + lane));
    if (instruction_flags & kDotHalfSelection) {
      // Select halves inside the arithmetic registers, without staging a wave.
      va = _mm_shuffle_epi8(va, _mm256_castsi256_si128(select_a));
      vb = _mm_shuffle_epi8(vb, _mm256_castsi256_si128(select_b));
    }
    auto vc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(c + lane));
    auto low = [](__m128i v) {
      if constexpr (Unsigned) {
        return _mm_and_si128(v, _mm_set1_epi32(0xffff));
      } else {
        return _mm_srai_epi32(_mm_slli_epi32(v, 16), 16);
      }
    };
    auto high = [](__m128i v) {
      if constexpr (Unsigned) {
        return _mm_srli_epi32(v, 16);
      } else {
        return _mm_srai_epi32(v, 16);
      }
    };
    auto widen = [](__m128i v) {
      if constexpr (Unsigned) {
        return _mm256_cvtepu32_epi64(v);
      } else {
        return _mm256_cvtepi32_epi64(v);
      }
    };
    auto product = [](__m256i x, __m256i y) {
      if constexpr (Unsigned) {
        return _mm256_mul_epu32(x, y);
      } else {
        return _mm256_mul_epi32(x, y);
      }
    };
    auto sum =
        _mm256_add_epi64(widen(vc), _mm256_add_epi64(product(widen(low(va)), widen(low(vb))),
                                                     product(widen(high(va)), widen(high(vb)))));
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      const auto upper = _mm256_set1_epi64x(Unsigned ? int64_t(UINT32_MAX) : INT32_MAX);
      sum = _mm256_blendv_epi8(sum, upper, _mm256_cmpgt_epi64(sum, upper));
      if constexpr (!Unsigned) {
        const auto lower = _mm256_set1_epi64x(INT32_MIN);
        sum = _mm256_blendv_epi8(sum, lower, _mm256_cmpgt_epi64(lower, sum));
      }
    }
    auto result = _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(sum, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
    auto lane_exec_mask =
        _mm_sllv_epi32(_mm_set1_epi32(int(exec_mask >> lane)), _mm_setr_epi32(31, 30, 29, 28));
    _mm_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Accumulate packed integer dot products. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void integer_dot_x86_64_v3(int bits, bool unsigned_acc, uint32_t exec_mask,
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
