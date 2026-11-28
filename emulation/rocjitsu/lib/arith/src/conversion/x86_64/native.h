// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native conversion instructions with the existing guest source/output policies.
// ISA-local names keep AVX2 and AVX-512 inline definitions separate.
#pragma once

#include "common/isa.h"
#include "conversion/conversion32.h"
#include "conversion/conversion64.h"
#include "conversion/float_conversion.h"
#include "conversion/model/conversion.h"
#include "conversion/x86_64/lanes.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd {

// Narrow one vector with the selected instruction-local rounding immediate.
template <int Rounding> inline Lanes<32> round_to_half(Lanes<32> raw) {
#if defined(__AVX512F__)
  return Lanes<32>(_mm512_cvtepu16_epi32(
      _mm512_cvt_roundps_ph(_mm512_castsi512_ps(raw.bits), Rounding | _MM_FROUND_NO_EXC)));
#else
  return Lanes<32>(_mm256_cvtepu16_epi32(
      _mm256_cvtps_ph(_mm256_castsi256_ps(raw.bits), Rounding | _MM_FROUND_NO_EXC)));
#endif
}

// Honor guest rounding without changing the host rounding mode.
inline Lanes<32> round_to_half(Lanes<32> raw, uint64_t flags) {
  switch (flags & RJ_ARITH_FP_ROUND_MASK) {
  case RJ_ARITH_FP_ROUND_UP:
    return round_to_half<_MM_FROUND_TO_POS_INF>(raw);
  case RJ_ARITH_FP_ROUND_DOWN:
    return round_to_half<_MM_FROUND_TO_NEG_INF>(raw);
  case RJ_ARITH_FP_ROUND_ZERO:
    return round_to_half<_MM_FROUND_TO_ZERO>(raw);
  default:
    return round_to_half<_MM_FROUND_TO_NEAREST_INT>(raw);
  }
}

// Flush according to the model's unbounded-exponent destination rounding,
// which can classify a value as tiny even when native subnormal rounding gives minimum normal.
template <typename From, typename To, typename V>
inline V flush_tiny_result(V source, V result, uint64_t flags) {
  using L = typename From::Lane;
  namespace model = conversion_model;
  constexpr L minimum_normal = L((From::kExponentMax >> 1) + 1 - (To::kExponentMax >> 1))
                               << From::kMantissaBits;
  constexpr int discarded = From::kMantissaBits - To::kMantissaBits;
  V threshold(minimum_normal);
  const auto direction = flags & RJ_ARITH_FP_ROUND_MASK;
  if (!direction) {
    threshold = V(minimum_normal - (L{1} << (discarded - 1)));
  } else if (direction != RJ_ARITH_FP_ROUND_ZERO) {
    const auto negative = (source & V(From::kSign)) != V(L{0});
    const auto away = direction == RJ_ARITH_FP_ROUND_DOWN ? negative : !negative;
    threshold = model::lane::choose(away, V(minimum_normal - (L{1} << discarded) + 1), threshold);
  }
  return model::lane::choose((source & V(From::kMagnitude)) < threshold, result & V(L(To::kSign)),
                             result);
}

// Apply source controls, round once to FP16, then finalize destination controls.
inline Lanes<32> float_to_half(Lanes<32> raw, uint64_t flags, uint32_t mode) {
  using V = Lanes<32>;
  namespace model = conversion_model;
  namespace fmt = model::fp_format;
  const auto policy = conversion_detail::float_policy<fmt::F32, fmt::F16>(flags, mode);
  raw = model::source_modifier::apply<fmt::F32>(raw, 0, policy.abs, policy.neg);
  raw = model::input_denormal::flush_input<fmt::F32>(raw, policy.input);
  V result = round_to_half(raw, flags);
  if (policy.rounding.saturate) {
    const auto overflow =
        (result & V(0x7fffU)) == V(0x7c00U) && (raw & V(0x7fffffffU)) < V(0x7f800000U);
    result = model::lane::choose(overflow, result - V(1U), result);
  }
  if (policy.rounding.flush_tiny) {
    result = flush_tiny_result<fmt::F32, fmt::F16>(raw, result, flags);
  }
  return model::output_modifier::apply<fmt::F16>(result, policy.output);
}

// Convert 16-bit integers exactly to FP32 before the single FP16 rounding.
template <bool Unsigned>
inline Lanes<32> integer_to_half(Lanes<32> raw, uint64_t flags, uint32_t mode) {
  using V = Lanes<32>;
  raw = (raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & V(0xffffU);
  if constexpr (!Unsigned) {
    raw = (raw ^ V(0x8000U)) - V(0x8000U);
  }
#if defined(__AVX512F__)
  V value(_mm512_castps_si512(_mm512_cvtepi32_ps(raw.bits)));
#else
  V value(_mm256_castps_si256(_mm256_cvtepi32_ps(raw.bits)));
#endif
  return float_to_half(value, flags, mode);
}

#if defined(__AVX512F__)
// Round a 32-bit integer vector directly to FP32 using AVX-512 local controls.
template <bool Unsigned, int Rounding> inline Lanes<32> round_integer(Lanes<32> raw) {
  if constexpr (Unsigned) {
    return Lanes<32>(
        _mm512_castps_si512(_mm512_cvt_roundepu32_ps(raw.bits, Rounding | _MM_FROUND_NO_EXC)));
  } else {
    return Lanes<32>(
        _mm512_castps_si512(_mm512_cvt_roundepi32_ps(raw.bits, Rounding | _MM_FROUND_NO_EXC)));
  }
}
#endif

// Convert signed/unsigned 32-bit integers, preserving directed guest rounding and OMOD.
template <bool Unsigned>
inline Lanes<32> integer_to_float(Lanes<32> raw, uint64_t flags, uint32_t mode) {
  using V = Lanes<32>;
  namespace model = conversion_model;
  namespace fmt = model::fp_format;
  V result;
#if defined(__AVX512F__)
  switch (flags & RJ_ARITH_FP_ROUND_MASK) {
  case RJ_ARITH_FP_ROUND_UP:
    result = round_integer<Unsigned, _MM_FROUND_TO_POS_INF>(raw);
    break;
  case RJ_ARITH_FP_ROUND_DOWN:
    result = round_integer<Unsigned, _MM_FROUND_TO_NEG_INF>(raw);
    break;
  case RJ_ARITH_FP_ROUND_ZERO:
    result = round_integer<Unsigned, _MM_FROUND_TO_ZERO>(raw);
    break;
  default:
    result = round_integer<Unsigned, _MM_FROUND_TO_NEAREST_INT>(raw);
    break;
  }
#else
  if (flags & RJ_ARITH_FP_ROUND_MASK) {
    // AVX2 integer conversion has no local rounding override. The shared vector
    // model handles directed modes; host rounding remains the default throughout.
    constexpr auto op = Unsigned ? Conversion32::UnsignedToFloat : Conversion32::SignedToFloat;
    return conversion_detail::convert32<op>(raw, flags, mode);
  }
  if constexpr (Unsigned) {
    // Widen exactly to FP64 before the single FP32 rounding.
    auto widen = [](__m128i words) {
      auto value = _mm256_cvtepi32_pd(words);
      auto high_bit = _mm256_cmp_pd(value, _mm256_setzero_pd(), _CMP_LT_OQ);
      return _mm256_add_pd(value, _mm256_and_pd(high_bit, _mm256_set1_pd(0x1p32)));
    };
    auto low = widen(_mm256_castsi256_si128(raw.bits));
    auto high = widen(_mm256_extracti128_si256(raw.bits, 1));
    auto value = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(low)),
                                      _mm256_cvtpd_ps(high), 1);
    result = V(_mm256_castps_si256(value));
  } else {
    result = V(_mm256_castps_si256(_mm256_cvtepi32_ps(raw.bits)));
  }
#endif
  const auto policy = conversion_detail::float_policy<
      std::conditional_t<Unsigned, model::conversion::U32, model::conversion::I32>, fmt::F32>(flags,
                                                                                              mode);
  return model::output_modifier::apply<fmt::F32>(result, policy.output);
}

// Widen FP32 or 32-bit integers exactly, retaining the inherited NaN and output policies.
template <Conversion64 Op>
inline Lanes<64> to_double(Lanes<64> raw, uint64_t flags, uint32_t mode, bool quiet_nan) {
  using V = Lanes<64>;
  namespace model = conversion_model;
  namespace fmt = model::fp_format;
  if constexpr (Op == Conversion64::FloatToDouble) {
    using Source = fmt::Format<uint64_t, 8, 23>;
    raw = model::source_modifier::apply<Source>(raw, bool(mode & RJ_ARITH_ALU_ABS_A),
                                                bool(mode & RJ_ARITH_ALU_NEG_A));
    raw = model::input_denormal::flush_input<Source>(
        raw, {bool(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)});
  }
  const auto words = raw.low_words();
  V result;
#if defined(__AVX512F__)
  if constexpr (Op == Conversion64::FloatToDouble) {
    result = V(_mm512_castpd_si512(_mm512_cvtps_pd(_mm256_castsi256_ps(words))));
  } else if constexpr (Op == Conversion64::UnsignedToDouble) {
    result = V(_mm512_castpd_si512(_mm512_cvtepu32_pd(words)));
  } else {
    result = V(_mm512_castpd_si512(_mm512_cvtepi32_pd(words)));
  }
#else
  if constexpr (Op == Conversion64::FloatToDouble) {
    result = V(_mm256_castpd_si256(_mm256_cvtps_pd(_mm_castsi128_ps(words))));
  } else {
    auto value = _mm256_cvtepi32_pd(words);
    if constexpr (Op == Conversion64::UnsignedToDouble) {
      auto high_bit = _mm256_cmp_pd(value, _mm256_setzero_pd(), _CMP_LT_OQ);
      value = _mm256_add_pd(value, _mm256_and_pd(high_bit, _mm256_set1_pd(0x1p32)));
    }
    result = V(_mm256_castpd_si256(value));
  }
#endif
  if constexpr (Op == Conversion64::FloatToDouble) {
    if (!quiet_nan) {
      const auto signaling =
          (raw & V(0x7fffffffULL)) > V(0x7f800000ULL) && (raw & V(0x400000ULL)) == V(uint64_t{0});
      result = model::lane::choose(signaling, result & V(~(1ULL << 51)), result);
    }
  }
  const auto policy = conversion_detail::float_policy<fmt::F32, fmt::F64>(flags, mode, quiet_nan);
  return model::output_modifier::apply<fmt::F64>(result, policy.output);
}

// Narrow FP64 lanes to FP32 with one instruction-local rounding on AVX-512.
template <int Rounding> inline Lanes<64> round_double_to_float(Lanes<64> raw) {
#if defined(__AVX512F__)
  auto value = _mm512_cvt_roundpd_ps(_mm512_castsi512_pd(raw.bits), Rounding | _MM_FROUND_NO_EXC);
  return Lanes<64>(_mm512_cvtepu32_epi64(_mm256_castps_si256(value)));
#else
  static_assert(Rounding == _MM_FROUND_TO_NEAREST_INT);
  auto value = _mm256_cvtpd_ps(_mm256_castsi256_pd(raw.bits));
  return Lanes<64>(_mm256_cvtepu32_epi64(_mm_castps_si128(value)));
#endif
}

// Narrow FP64 to FP32 with inherited source, tininess, NaN and output policies.
inline Lanes<64> double_to_float(Lanes<64> raw, uint64_t flags, uint32_t mode) {
  using V = Lanes<64>;
  namespace model = conversion_model;
  namespace fmt = model::fp_format;
#if !defined(__AVX512F__)
  if (flags & RJ_ARITH_FP_ROUND_MASK) {
    return conversion_detail::convert64<Conversion64::DoubleToFloat>(raw, flags, mode, true);
  }
#endif
  const auto policy = conversion_detail::float_policy<fmt::F64, fmt::F32>(flags, mode);
  raw = model::source_modifier::apply<fmt::F64>(raw, 0, policy.abs, policy.neg);
  raw = model::input_denormal::flush_input<fmt::F64>(raw, policy.input);
  V result;
#if defined(__AVX512F__)
  switch (flags & RJ_ARITH_FP_ROUND_MASK) {
  case RJ_ARITH_FP_ROUND_UP:
    result = round_double_to_float<_MM_FROUND_TO_POS_INF>(raw);
    break;
  case RJ_ARITH_FP_ROUND_DOWN:
    result = round_double_to_float<_MM_FROUND_TO_NEG_INF>(raw);
    break;
  case RJ_ARITH_FP_ROUND_ZERO:
    result = round_double_to_float<_MM_FROUND_TO_ZERO>(raw);
    break;
  default:
    result = round_double_to_float<_MM_FROUND_TO_NEAREST_INT>(raw);
    break;
  }
#else
  result = round_double_to_float<_MM_FROUND_TO_NEAREST_INT>(raw);
#endif
  if (policy.rounding.flush_tiny) {
    result = flush_tiny_result<fmt::F64, fmt::F32>(raw, result, flags);
  }
  using Destination = fmt::Format<uint64_t, 8, 23>;
  return model::output_modifier::apply<Destination>(result, policy.output);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::conversion_simd
