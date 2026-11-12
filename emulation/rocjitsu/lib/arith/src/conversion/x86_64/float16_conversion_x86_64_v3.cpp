// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16/BF16 conversions preserving inherited NaN, subnormal and overflow rules.
#include "conversion/float16_conversion.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace {

// Select lanes without arithmetic on exceptional floating-point representations.
__m256i select(__m256i condition, __m256i yes, __m256i no) {
  return _mm256_blendv_epi8(no, yes, condition);
}

// Narrow using rocjitsu's inherited RNE/SR representations and finite saturation rule.
__m256i narrow_bf16(__m256i raw, __m256i seed, bool sr, bool saturate) {
  auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  auto inf = _mm256_set1_epi32(0x7f800000);
  auto sign = _mm256_srli_epi32(raw, 16);
  __m256i result;
  if (sr) {
    // BF16 retains the FP32 exponent, including subnormal inputs.
    result = _mm256_srli_epi32(_mm256_add_epi32(raw, _mm256_srli_epi32(seed, 16)), 16);
    result = select(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()), sign, result);
    result = select(_mm256_cmpeq_epi32(magnitude, inf), sign, result);
    result = select(_mm256_cmpgt_epi32(magnitude, inf),
                    _mm256_or_si256(sign, _mm256_set1_epi32(0x7fff)), result);
  } else {
    auto odd = _mm256_and_si256(sign, _mm256_set1_epi32(1));
    auto rounded = _mm256_add_epi32(raw, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), odd));
    auto tail = _mm256_and_si256(raw, _mm256_set1_epi32(0xffff));
    auto special = select(_mm256_cmpgt_epi32(tail, _mm256_setzero_si256()),
                          _mm256_or_si256(sign, _mm256_set1_epi32(1)), sign);
    result = select(_mm256_cmpgt_epi32(inf, magnitude), _mm256_srli_epi32(rounded, 16), special);
  }
  if (saturate) {
    auto overflow = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(0x7f80));
    auto clipped = _mm256_sub_epi32(result, _mm256_set1_epi32(1));
    result = select(_mm256_cmpgt_epi32(inf, magnitude), select(overflow, clipped, result), result);
  }
  return result;
}

// Narrow FP32 to FP16 with inherited NaN payloads or seeded stochastic rounding.
__m256i narrow_f16(__m256i raw, __m256i seed, bool sr, bool saturate) {
  auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  auto sign = _mm256_and_si256(_mm256_srli_epi32(raw, 16), _mm256_set1_epi32(0x8000));
  auto inf = _mm256_set1_epi32(0x7f800000);
  __m256i result;
  if (sr) {
    auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127));
    auto mantissa = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffff));
    auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(-14), exponent);
    // Preserve the inherited alignment-before-random-increment convention.
    auto shifted = _mm256_srlv_epi32(_mm256_or_si256(mantissa, _mm256_set1_epi32(0x800000)),
                                     _mm256_sub_epi32(_mm256_set1_epi32(-14), exponent));
    mantissa = select(subnormal, shifted, mantissa);
    exponent = select(subnormal, _mm256_setzero_si256(),
                      _mm256_add_epi32(exponent, _mm256_set1_epi32(15)));
    mantissa = _mm256_add_epi32(mantissa, _mm256_srli_epi32(seed, 19));
    auto carry = _mm256_srli_epi32(mantissa, 23);
    exponent = _mm256_add_epi32(exponent, carry);
    result = _mm256_or_si256(
        _mm256_slli_epi32(exponent, 10),
        _mm256_and_si256(_mm256_srli_epi32(mantissa, 13), _mm256_set1_epi32(0x3ff)));
    result = select(_mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(30)), _mm256_set1_epi32(0x7c00),
                    result);
    result = _mm256_or_si256(result, sign);
    result = select(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()), sign, result);
    result = select(_mm256_cmpeq_epi32(magnitude, inf),
                    _mm256_or_si256(sign, _mm256_set1_epi32(0x7c00)), result);
    result = select(_mm256_cmpgt_epi32(magnitude, inf),
                    _mm256_or_si256(sign, _mm256_set1_epi32(0x7fff)), result);
  } else {
    // F16C/AVX-512 provide fixed RNE, but quiet NaNs. Restore the inherited payload.
    result = _mm256_cvtepu16_epi32(
        _mm256_cvtps_ph(_mm256_castsi256_ps(raw), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    auto payload = _mm256_and_si256(_mm256_srli_epi32(raw, 13), _mm256_set1_epi32(0x3ff));
    payload =
        select(_mm256_cmpeq_epi32(payload, _mm256_setzero_si256()), _mm256_set1_epi32(1), payload);
    result =
        select(_mm256_cmpgt_epi32(magnitude, inf),
               _mm256_or_si256(sign, _mm256_or_si256(_mm256_set1_epi32(0x7c00), payload)), result);
  }
  if (saturate) {
    auto overflow = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(0x7c00));
    result =
        select(_mm256_cmpgt_epi32(inf, magnitude),
               select(overflow, _mm256_sub_epi32(result, _mm256_set1_epi32(1)), result), result);
  }
  return result;
}

// Share packing and dispatch while keeping each format's numerical model explicit.
__m256i narrow(rocjitsu::arith::Float16Format format, __m256i raw, __m256i seed, bool sr,
               bool saturate) {
  return format == rocjitsu::arith::Float16Format::F16 ? narrow_f16(raw, seed, sr, saturate)
                                                       : narrow_bf16(raw, seed, sr, saturate);
}

// Apply source sign modifiers without changing NaN payloads.
__m256i source_sign(__m256i raw, uint32_t mode, int operand) {
  if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
    raw = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  }
  if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
    raw = _mm256_xor_si256(raw, _mm256_set1_epi32(int(0x80000000U)));
  }
  return raw;
}

// Match the packed instruction's independent half-word addends.
__m256i narrow_packed_sr(rocjitsu::arith::Float16Format format, __m256i raw, __m256i addend,
                         bool saturate) {
  if (format == rocjitsu::arith::Float16Format::F16) {
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
    raw = select(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x47800000), magnitude),
                 _mm256_add_epi32(raw, addend), raw);
    return narrow_f16(raw, _mm256_setzero_si256(), true, saturate);
  }
  return narrow_bf16(raw, _mm256_slli_epi32(addend, 16), true, saturate);
}

} // namespace

namespace rocjitsu::arith {

// Convert a Wave32 with validated controls; reads precede masked stores, including aliases.
void float16_conversion_x86_64_v3(Float16Format format, Float16Conversion op, uint64_t flags,
                                  uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b, const uint32_t *c) {
  bool saturate = flags & RJ_ARITH_FP16_OVFL;
  for (int lane = 0; lane < 32; lane += 8) {
    auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i value;
    if (op == Float16Conversion::Widen) {
      value = mode & RJ_ARITH_ALU_HIGH_A ? _mm256_and_si256(av, _mm256_set1_epi32(int(0xffff0000U)))
                                         : _mm256_slli_epi32(av, 16);
    } else {
      auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
      if (op == Float16Conversion::Stochastic) {
        value = narrow(format, source_sign(av, mode, 0), bv, true, saturate);
        auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
        bool high = mode & RJ_ARITH_ALU_HIGH_D;
        auto keep = _mm256_set1_epi32(high ? 0xffff : int(0xffff0000U));
        value = _mm256_or_si256(_mm256_and_si256(old, keep),
                                high ? _mm256_slli_epi32(value, 16) : value);
      } else {
        if (op == Float16Conversion::StochasticPack) {
          auto seed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
          auto low = _mm256_and_si256(seed, _mm256_set1_epi32(0xffff));
          auto high = _mm256_srli_epi32(seed, 16);
          value =
              _mm256_or_si256(narrow_packed_sr(format, av, low, saturate),
                              _mm256_slli_epi32(narrow_packed_sr(format, bv, high, saturate), 16));
        } else {
          auto zero = _mm256_setzero_si256();
          value = _mm256_or_si256(
              narrow(format, source_sign(av, mode, 0), zero, false, saturate),
              _mm256_slli_epi32(narrow(format, source_sign(bv, mode, 1), zero, false, saturate),
                                16));
        }
      }
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

} // namespace rocjitsu::arith
