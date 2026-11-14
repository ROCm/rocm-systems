// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 division operand rescaling and its per-lane scaling-condition output.
// CPU-specific arithmetic stays here; public entry points select this backend.

// Vectorized form of the rocjitsu-derived division pre-scaling model.

#include "fp/div_scale_impl.h"
#include "fp/float_output.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> class Ops {
public:
  using V = __m256i;
  using Mask = __m256i;

  // Broadcast the supplied bit pattern to each vector lane.
  static V set(uint64_t x) {
    if constexpr (Wide) {
      return _mm256_set1_epi64x(int64_t(x));
    } else {
      return _mm256_set1_epi32(int32_t(x));
    }
  }

  // Add corresponding integer lanes with unsigned wraparound.
  static V add(V a, V b) {
    if constexpr (Wide) {
      return _mm256_add_epi64(a, b);
    } else {
      return _mm256_add_epi32(a, b);
    }
  }

  // Subtract corresponding integer lanes with unsigned wraparound.
  static V sub(V a, V b) {
    if constexpr (Wide) {
      return _mm256_sub_epi64(a, b);
    } else {
      return _mm256_sub_epi32(a, b);
    }
  }

  // Shift each unsigned lane right by the supplied count.
  static V shr(V a, int count) {
    if constexpr (Wide) {
      return _mm256_srli_epi64(a, count);
    } else {
      return _mm256_srli_epi32(a, count);
    }
  }

  // Shift each lane left by the supplied count.
  static V shl(V a, int count) {
    if constexpr (Wide) {
      return _mm256_slli_epi64(a, count);
    } else {
      return _mm256_slli_epi32(a, count);
    }
  }

  // Return the lane mask for signed a > b.
  static Mask gt(V a, V b) {
    if constexpr (Wide) {
      return _mm256_cmpgt_epi64(a, b);
    } else {
      return _mm256_cmpgt_epi32(a, b);
    }
  }

  // Return the lane mask for equal operands.
  static Mask eq(V a, V b) {
    if constexpr (Wide) {
      return _mm256_cmpeq_epi64(a, b);
    } else {
      return _mm256_cmpeq_epi32(a, b);
    }
  }

  // Intersect lane predicates.
  static Mask both(Mask a, Mask b) { return _mm256_and_si256(a, b); }

  // Unite lane predicates.
  static Mask either(Mask a, Mask b) { return _mm256_or_si256(a, b); }

  // Complement a lane predicate.
  static Mask inverse(Mask a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

  // Select values lane by lane according to the supplied predicate.
  static V select(Mask m, V yes, V no) { return _mm256_blendv_epi8(no, yes, m); }
};

// Return scaled division operands and produce the post-scaling predicate for DIV_FMAS.
template <int Width>
__m256i scale_value(__m256i a, __m256i b, __m256i c, uint32_t mode, bool quiet_nan, uint64_t flags,
                    typename Ops<Width == 64>::Mask &post) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using VectorOps = Ops<Width == 64>;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  auto zero = VectorOps::set(0), one = VectorOps::set(1), inf = VectorOps::set(F::infinity),
       signbit = VectorOps::set(F::sign), magmask = VectorOps::set(F::sign - 1);
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    a = VectorOps::select(VectorOps::eq(_mm256_and_si256(a, inf), zero),
                          _mm256_and_si256(a, signbit), a);
    b = VectorOps::select(VectorOps::eq(_mm256_and_si256(b, inf), zero),
                          _mm256_and_si256(b, signbit), b);
    c = VectorOps::select(VectorOps::eq(_mm256_and_si256(c, inf), zero),
                          _mm256_and_si256(c, signbit), c);
  }
  auto mag = _mm256_and_si256(a, magmask), bm = _mm256_and_si256(b, magmask),
       cm = _mm256_and_si256(c, magmask);
  auto de = VectorOps::shr(_mm256_and_si256(b, inf), F::fraction),
       ne = VectorOps::shr(_mm256_and_si256(c, inf), F::fraction);
  auto delta = VectorOps::sub(ne, de);
  auto denominator = VectorOps::eq(a, b);
  auto large_delta = VectorOps::gt(delta, VectorOps::set(threshold - 1)),
       small_delta = VectorOps::gt(VectorOps::set(uint64_t(-int64_t(threshold) + 1)), delta);
  post = VectorOps::either(large_delta, small_delta);
  auto adjustment = VectorOps::select(VectorOps::gt(VectorOps::set(F::fraction + 2), ne),
                                      VectorOps::set(scale), zero);
  adjustment = VectorOps::select(
      small_delta, VectorOps::select(denominator, zero, VectorOps::set(scale)), adjustment);
  auto high_denominator = VectorOps::gt(de, VectorOps::set(2 * F::bias - 2));
  adjustment = VectorOps::select(
      high_denominator,
      VectorOps::select(VectorOps::both(small_delta, VectorOps::inverse(denominator)), zero,
                        VectorOps::set(uint64_t(-int64_t(scale)))),
      adjustment);
  adjustment = VectorOps::select(VectorOps::eq(de, zero), VectorOps::set(scale), adjustment);
  adjustment = VectorOps::select(
      large_delta, VectorOps::select(denominator, VectorOps::set(scale), zero), adjustment);
  auto fraction_mask = VectorOps::set((uint64_t(1) << F::fraction) - 1),
       hidden = VectorOps::set(uint64_t(1) << F::fraction);
  auto fraction = _mm256_and_si256(a, fraction_mask);

  // Encoding 2^fraction + significand then subtracting 2^fraction converts
  // the integer significand exactly, without touching subnormal FP operands.
  // The result supplies a normalized mantissa and its leading-bit exponent.
  auto magic = VectorOps::set(uint64_t(F::bias + F::fraction) << F::fraction);
  auto encoded = _mm256_or_si256(magic, fraction);
  __m256i normalized;
  if constexpr (Width == 64) {
    normalized =
        _mm256_castpd_si256(_mm256_sub_pd(_mm256_castsi256_pd(encoded), _mm256_set1_pd(0x1p52)));
  } else {
    normalized =
        _mm256_castps_si256(_mm256_sub_ps(_mm256_castsi256_ps(encoded), _mm256_set1_ps(0x1p23f)));
  }
  auto original_exponent = VectorOps::shr(_mm256_and_si256(a, inf), F::fraction);
  auto subnormal = VectorOps::eq(original_exponent, zero);
  auto small_exponent =
      VectorOps::sub(VectorOps::shr(_mm256_and_si256(normalized, inf), F::fraction),
                     VectorOps::set(F::bias + F::fraction - 1));
  auto exponent =
      VectorOps::add(VectorOps::select(subnormal, small_exponent, original_exponent), adjustment);
  auto mantissa = _mm256_and_si256(VectorOps::select(subnormal, normalized, a), fraction_mask);
  auto sign = _mm256_and_si256(a, signbit);
  auto result =
      _mm256_or_si256(sign, _mm256_or_si256(VectorOps::shl(exponent, F::fraction), mantissa));
  auto shift = VectorOps::sub(one, exponent);
  shift = VectorOps::select(VectorOps::gt(one, shift), one, shift);
  shift = VectorOps::select(VectorOps::gt(shift, VectorOps::set(F::fraction + 2)),
                            VectorOps::set(F::fraction + 2), shift);
  auto significand = _mm256_or_si256(mantissa, hidden);
  __m256i half, kept, rounded;
  if constexpr (Width == 64) {
    half = _mm256_sllv_epi64(one, VectorOps::sub(shift, one));
    kept = _mm256_srlv_epi64(significand, shift);
    rounded =
        _mm256_srlv_epi64(VectorOps::add(significand, VectorOps::add(VectorOps::sub(half, one),
                                                                     _mm256_and_si256(kept, one))),
                          shift);
  } else {
    half = _mm256_sllv_epi32(one, VectorOps::sub(shift, one));
    kept = _mm256_srlv_epi32(significand, shift);
    rounded =
        _mm256_srlv_epi32(VectorOps::add(significand, VectorOps::add(VectorOps::sub(half, one),
                                                                     _mm256_and_si256(kept, one))),
                          shift);
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto overflow_limit = inf;
  if (rounding) {
    auto away = rounding == 1   ? VectorOps::eq(sign, zero)
                : rounding == 2 ? VectorOps::inverse(VectorOps::eq(sign, zero))
                                : VectorOps::eq(one, zero);

    auto discarded = _mm256_and_si256(significand, VectorOps::sub(VectorOps::add(half, half), one));
    auto increment = VectorOps::both(away, VectorOps::inverse(VectorOps::eq(discarded, zero)));
    rounded = VectorOps::add(kept, VectorOps::select(increment, one, zero));
    overflow_limit = VectorOps::select(away, inf, VectorOps::sub(inf, one));
  }
  result = VectorOps::select(VectorOps::gt(one, exponent), _mm256_or_si256(sign, rounded), result);
  result = VectorOps::select(VectorOps::gt(exponent, VectorOps::set(2 * F::bias)),
                             _mm256_or_si256(sign, overflow_limit), result);
  auto quiet =
      VectorOps::select(VectorOps::gt(mag, inf), VectorOps::set(quiet_nan ? F::quiet : 0), zero);
  result = VectorOps::select(
      VectorOps::either(VectorOps::eq(mag, zero), VectorOps::inverse(VectorOps::gt(inf, mag))),
      _mm256_or_si256(a, quiet), result);
  result = VectorOps::select(VectorOps::either(VectorOps::eq(bm, zero), VectorOps::eq(cm, zero)),
                             VectorOps::set(F::sign | F::infinity | F::quiet), result);
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    result = VectorOps::select(VectorOps::eq(_mm256_and_si256(result, inf), zero),
                               _mm256_and_si256(result, signbit), result);
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    auto output_negative =
        VectorOps::inverse(VectorOps::eq(_mm256_and_si256(result, signbit), zero));
    auto output_away = rounding == 1 ? VectorOps::inverse(output_negative) : output_negative;
    auto overflow = !rounding       ? inf
                    : rounding == 3 ? VectorOps::sub(inf, one)
                                    : VectorOps::select(output_away, inf, VectorOps::sub(inf, one));

    auto exponent = _mm256_and_si256(result, inf), output_sign = _mm256_and_si256(result, signbit),
         step = VectorOps::set(uint64_t(1) << F::fraction);
    __m256i scaled;
    if (omod == 3) {
      scaled = VectorOps::select(VectorOps::eq(exponent, step), output_sign,
                                 VectorOps::sub(result, step));
    } else {
      auto increment = VectorOps::set(uint64_t(omod) << F::fraction);
      scaled = VectorOps::select(
          VectorOps::inverse(VectorOps::gt(VectorOps::sub(inf, increment), exponent)),
          _mm256_or_si256(output_sign, overflow), VectorOps::add(result, increment));
    }
    result = VectorOps::select(VectorOps::eq(exponent, inf), result, scaled);
    result = VectorOps::select(VectorOps::eq(exponent, zero), zero, result);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const auto unclamped = result;
    auto negative = VectorOps::inverse(VectorOps::eq(_mm256_and_si256(result, signbit), zero));
    result =
        VectorOps::select(VectorOps::either(negative, VectorOps::gt(result, inf)), zero, result);
    auto one = VectorOps::set(uint64_t(F::bias) << F::fraction);
    result = VectorOps::select(VectorOps::gt(result, one), one, result);
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      result = VectorOps::select(
          VectorOps::gt(_mm256_and_si256(unclamped, VectorOps::set(F::sign - 1)), inf), unclamped,
          result);
    }
  }
  return result;
}

// Load source values in the vector layout expected by this kernel.
template <int Width> __m256i load(const uint32_t *const *v, int lane, uint32_t mode, int operand) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using VectorOps = Ops<Width == 64>;
  __m256i result;
  if constexpr (Width == 64) {
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[1] + lane)));
    result = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
  } else {
    result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[0] + lane));
  }
  auto flip = VectorOps::set(mode & (RJ_ARITH_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm256_xor_si256(result, flip);
}

} // namespace

namespace rocjitsu::arith {

// Rescale division operands and produce scaling conditions. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
uint32_t div_scale_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, bool quiet_nan, uint64_t flags) {
  constexpr int lanes = Width == 64 ? 4 : 8;
  uint32_t staged[2][32], conditions = 0;
  for (int lane = 0; lane < 32; lane += lanes) {
    typename Ops<Width == 64>::Mask post;
    auto result = scale_value<Width>(load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1),
                                     load<Width>(c, lane, mode, 2), mode, quiet_nan, flags, post);
    if constexpr (Width == 64) {
      conditions |= uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(post))) << lane;
    } else {
      conditions |= uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(post))) << lane;
    }
    if constexpr (Width == 64) {
      auto words = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[0] + lane),
                       _mm256_castsi256_si128(words));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[1] + lane),
                       _mm256_extracti128_si256(words, 1));
    } else {
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), lane_exec_mask, result);
    }
  }
  // Preserve cross-half aliases by committing D0 then D1 after all reads.
  if constexpr (Width == 64) {
    for (int reg = 0; reg < 2; ++reg) {
      for (int lane = 0; lane < 32; lane += 8) {
        auto result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane));
        auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                                _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
        _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask, result);
      }
    }
  }
  return conditions & exec_mask;
}

template uint32_t div_scale_x86_64_v3<32>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *, bool, uint64_t);
template uint32_t div_scale_x86_64_v3<64>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *, bool, uint64_t);

} // namespace rocjitsu::arith
