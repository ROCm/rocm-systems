// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 division operand rescaling and its per-lane scaling-condition output.
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
  using V = __m512i;
  using Mask = __mmask16;

  // Broadcast the supplied bit pattern to each vector lane.
  static V set(uint64_t x) {
    if constexpr (Wide) {
      return _mm512_set1_epi64(int64_t(x));
    } else {
      return _mm512_set1_epi32(int32_t(x));
    }
  }

  // Add corresponding integer lanes with unsigned wraparound.
  static V add(V a, V b) {
    if constexpr (Wide) {
      return _mm512_add_epi64(a, b);
    } else {
      return _mm512_add_epi32(a, b);
    }
  }

  // Subtract corresponding integer lanes with unsigned wraparound.
  static V sub(V a, V b) {
    if constexpr (Wide) {
      return _mm512_sub_epi64(a, b);
    } else {
      return _mm512_sub_epi32(a, b);
    }
  }

  // Shift each unsigned lane right by the supplied count.
  static V shr(V a, int count) {
    if constexpr (Wide) {
      return _mm512_srli_epi64(a, count);
    } else {
      return _mm512_srli_epi32(a, count);
    }
  }

  // Shift each lane left by the supplied count.
  static V shl(V a, int count) {
    if constexpr (Wide) {
      return _mm512_slli_epi64(a, count);
    } else {
      return _mm512_slli_epi32(a, count);
    }
  }

  // Return the lane mask for signed a > b.
  static Mask gt(V a, V b) {
    if constexpr (Wide) {
      return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT);
    } else {
      return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT);
    }
  }

  // Return the lane mask for equal operands.
  static Mask eq(V a, V b) {
    if constexpr (Wide) {
      return _mm512_cmpeq_epi64_mask(a, b);
    } else {
      return _mm512_cmpeq_epi32_mask(a, b);
    }
  }

  // Intersect lane predicates.
  static Mask both(Mask a, Mask b) { return a & b; }

  // Unite lane predicates.
  static Mask either(Mask a, Mask b) { return a | b; }

  // Complement a lane predicate.
  static Mask inverse(Mask a) { return Mask(~a); }

  // Select values lane by lane according to the supplied predicate.
  static V select(Mask m, V yes, V no) {
    if constexpr (Wide) {
      return _mm512_mask_blend_epi64(__mmask8(m), no, yes);
    } else {
      return _mm512_mask_blend_epi32(m, no, yes);
    }
  }
};

// Return scaled division operands and produce the post-scaling predicate for DIV_FMAS.
template <int Width>
__m512i scale_value(__m512i a, __m512i b, __m512i c, uint32_t mode, bool quiet_nan, uint64_t flags,
                    typename Ops<Width == 64>::Mask &post) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using VectorOps = Ops<Width == 64>;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  auto zero = VectorOps::set(0), one = VectorOps::set(1), inf = VectorOps::set(F::infinity),
       signbit = VectorOps::set(F::sign), magmask = VectorOps::set(F::sign - 1);
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    a = VectorOps::select(VectorOps::eq(_mm512_and_si512(a, inf), zero),
                          _mm512_and_si512(a, signbit), a);
    b = VectorOps::select(VectorOps::eq(_mm512_and_si512(b, inf), zero),
                          _mm512_and_si512(b, signbit), b);
    c = VectorOps::select(VectorOps::eq(_mm512_and_si512(c, inf), zero),
                          _mm512_and_si512(c, signbit), c);
  }
  auto mag = _mm512_and_si512(a, magmask), bm = _mm512_and_si512(b, magmask),
       cm = _mm512_and_si512(c, magmask);
  auto de = VectorOps::shr(_mm512_and_si512(b, inf), F::fraction),
       ne = VectorOps::shr(_mm512_and_si512(c, inf), F::fraction);
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
  auto fraction = _mm512_and_si512(a, fraction_mask);

  // Encoding 2^fraction + significand then subtracting 2^fraction converts
  // the integer significand exactly, without touching subnormal FP operands.
  // The result supplies a normalized mantissa and its leading-bit exponent.
  auto magic = VectorOps::set(uint64_t(F::bias + F::fraction) << F::fraction);
  auto encoded = _mm512_or_si512(magic, fraction);
  __m512i normalized;
  if constexpr (Width == 64) {
    normalized =
        _mm512_castpd_si512(_mm512_sub_pd(_mm512_castsi512_pd(encoded), _mm512_set1_pd(0x1p52)));
  } else {
    normalized =
        _mm512_castps_si512(_mm512_sub_ps(_mm512_castsi512_ps(encoded), _mm512_set1_ps(0x1p23f)));
  }
  auto original_exponent = VectorOps::shr(_mm512_and_si512(a, inf), F::fraction);
  auto subnormal = VectorOps::eq(original_exponent, zero);
  auto small_exponent =
      VectorOps::sub(VectorOps::shr(_mm512_and_si512(normalized, inf), F::fraction),
                     VectorOps::set(F::bias + F::fraction - 1));
  auto exponent =
      VectorOps::add(VectorOps::select(subnormal, small_exponent, original_exponent), adjustment);
  auto mantissa = _mm512_and_si512(VectorOps::select(subnormal, normalized, a), fraction_mask);
  auto sign = _mm512_and_si512(a, signbit);
  auto result =
      _mm512_or_si512(sign, _mm512_or_si512(VectorOps::shl(exponent, F::fraction), mantissa));
  auto shift = VectorOps::sub(one, exponent);
  shift = VectorOps::select(VectorOps::gt(one, shift), one, shift);
  shift = VectorOps::select(VectorOps::gt(shift, VectorOps::set(F::fraction + 2)),
                            VectorOps::set(F::fraction + 2), shift);
  auto significand = _mm512_or_si512(mantissa, hidden);
  __m512i half, kept, rounded;
  if constexpr (Width == 64) {
    half = _mm512_sllv_epi64(one, VectorOps::sub(shift, one));
    kept = _mm512_srlv_epi64(significand, shift);
    rounded =
        _mm512_srlv_epi64(VectorOps::add(significand, VectorOps::add(VectorOps::sub(half, one),
                                                                     _mm512_and_si512(kept, one))),
                          shift);
  } else {
    half = _mm512_sllv_epi32(one, VectorOps::sub(shift, one));
    kept = _mm512_srlv_epi32(significand, shift);
    rounded =
        _mm512_srlv_epi32(VectorOps::add(significand, VectorOps::add(VectorOps::sub(half, one),
                                                                     _mm512_and_si512(kept, one))),
                          shift);
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  auto overflow_limit = inf;
  if (rounding) {
    auto away = rounding == 1   ? VectorOps::eq(sign, zero)
                : rounding == 2 ? VectorOps::inverse(VectorOps::eq(sign, zero))
                                : VectorOps::eq(one, zero);

    auto discarded = _mm512_and_si512(significand, VectorOps::sub(VectorOps::add(half, half), one));
    auto increment = VectorOps::both(away, VectorOps::inverse(VectorOps::eq(discarded, zero)));
    rounded = VectorOps::add(kept, VectorOps::select(increment, one, zero));
    overflow_limit = VectorOps::select(away, inf, VectorOps::sub(inf, one));
  }
  result = VectorOps::select(VectorOps::gt(one, exponent), _mm512_or_si512(sign, rounded), result);
  result = VectorOps::select(VectorOps::gt(exponent, VectorOps::set(2 * F::bias)),
                             _mm512_or_si512(sign, overflow_limit), result);
  auto quiet =
      VectorOps::select(VectorOps::gt(mag, inf), VectorOps::set(quiet_nan ? F::quiet : 0), zero);
  result = VectorOps::select(
      VectorOps::either(VectorOps::eq(mag, zero), VectorOps::inverse(VectorOps::gt(inf, mag))),
      _mm512_or_si512(a, quiet), result);
  result = VectorOps::select(VectorOps::either(VectorOps::eq(bm, zero), VectorOps::eq(cm, zero)),
                             VectorOps::set(F::sign | F::infinity | F::quiet), result);
  if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
    result = VectorOps::select(VectorOps::eq(_mm512_and_si512(result, inf), zero),
                               _mm512_and_si512(result, signbit), result);
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    auto output_negative =
        VectorOps::inverse(VectorOps::eq(_mm512_and_si512(result, signbit), zero));
    auto output_away = rounding == 1 ? VectorOps::inverse(output_negative) : output_negative;
    auto overflow = !rounding       ? inf
                    : rounding == 3 ? VectorOps::sub(inf, one)
                                    : VectorOps::select(output_away, inf, VectorOps::sub(inf, one));

    auto exponent = _mm512_and_si512(result, inf), output_sign = _mm512_and_si512(result, signbit),
         step = VectorOps::set(uint64_t(1) << F::fraction);
    __m512i scaled;
    if (omod == 3) {
      scaled = VectorOps::select(VectorOps::eq(exponent, step), output_sign,
                                 VectorOps::sub(result, step));
    } else {
      auto increment = VectorOps::set(uint64_t(omod) << F::fraction);
      scaled = VectorOps::select(
          VectorOps::inverse(VectorOps::gt(VectorOps::sub(inf, increment), exponent)),
          _mm512_or_si512(output_sign, overflow), VectorOps::add(result, increment));
    }
    result = VectorOps::select(VectorOps::eq(exponent, inf), result, scaled);
    result = VectorOps::select(VectorOps::eq(exponent, zero), zero, result);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const auto unclamped = result;
    auto negative = VectorOps::inverse(VectorOps::eq(_mm512_and_si512(result, signbit), zero));
    result =
        VectorOps::select(VectorOps::either(negative, VectorOps::gt(result, inf)), zero, result);
    auto one = VectorOps::set(uint64_t(F::bias) << F::fraction);
    result = VectorOps::select(VectorOps::gt(result, one), one, result);
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      result = VectorOps::select(
          VectorOps::gt(_mm512_and_si512(unclamped, VectorOps::set(F::sign - 1)), inf), unclamped,
          result);
    }
  }
  return result;
}

// Load source values in the vector layout expected by this kernel.
template <int Width> __m512i load(const uint32_t *const *v, int lane, uint32_t mode, int operand) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using VectorOps = Ops<Width == 64>;
  __m512i result;
  if constexpr (Width == 64) {
    auto low =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[0] + lane)));
    auto high =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[1] + lane)));
    result = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
  } else {
    result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v[0] + lane));
  }
  auto flip = VectorOps::set(mode & (RJ_ARITH_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm512_xor_si512(result, flip);
}

} // namespace

namespace rocjitsu::arith {

// Rescale division operands and produce scaling conditions. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width>
uint32_t div_scale_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, bool quiet_nan, uint64_t flags) {
  constexpr int lanes = Width == 64 ? 8 : 16;
  uint32_t staged[2][32], conditions = 0;
  for (int lane = 0; lane < 32; lane += lanes) {
    typename Ops<Width == 64>::Mask post;
    auto result = scale_value<Width>(load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1),
                                     load<Width>(c, lane, mode, 2), mode, quiet_nan, flags, post);
    conditions |= uint32_t(post) << lane;
    if constexpr (Width == 64) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                          _mm512_cvtepi64_epi32(result));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
    } else {
      _mm512_mask_storeu_epi32(d[0] + lane, __mmask16(exec_mask >> lane), result);
    }
  }
  // Preserve cross-half aliases by committing D0 then D1 after all reads.
  if constexpr (Width == 64) {
    for (int reg = 0; reg < 2; ++reg) {
      for (int lane = 0; lane < 32; lane += 16) {
        auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
        _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane), result);
      }
    }
  }
  return conditions & exec_mask;
}

template uint32_t div_scale_x86_64_v4<32>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *, bool, uint64_t);
template uint32_t div_scale_x86_64_v4<64>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *, bool, uint64_t);

} // namespace rocjitsu::arith
