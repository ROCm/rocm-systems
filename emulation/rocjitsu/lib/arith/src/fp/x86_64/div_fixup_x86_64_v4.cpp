// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 division-result fixup, special values, and optional guest exception causes.
// CPU-specific arithmetic stays here; public entry points select this backend.

// Vectorized form of the rocjitsu-derived division fixup model.

#include "fp/div_fixup_impl.h"
#include "fp/float_output.h"
#include "fp/x86_64/div_fixup_exceptions.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> class Ops {
public:
  using V = __m512i;
  using Mask = __mmask16;

  // Collect vector lane predicates into a scalar bit mask.
  static uint32_t bits(Mask mask) { return mask; }

  // Broadcast the supplied bit pattern to each vector lane.
  static V set(uint64_t x) {
    if constexpr (Wide) {
      return _mm512_set1_epi64(int64_t(x));
    } else {
      return _mm512_set1_epi32(int32_t(x));
    }
  }

  // Return the bitwise intersection of the vector operands.
  static V band(V a, V b) { return _mm512_and_si512(a, b); }

  // Return the bitwise union of the vector operands.
  static V bor(V a, V b) { return _mm512_or_si512(a, b); }

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

// Return fixed-up division result bits from quotient A, denominator B, and numerator C, with
// format-specific special values and output modifiers.
template <int Width, bool Rdna3>
__m512i value(__m512i a, __m512i b, __m512i c, uint32_t mode, bool saturate, bool dx10_clamp_off,
              bool ieee) {
  using F = rocjitsu::arith::IeeeFormat<Width>;
  using VectorOps = Ops<Width == 64>;
  auto zero = VectorOps::set(0), inf = VectorOps::set(F::infinity),
       signbit = VectorOps::set(F::sign), magmask = VectorOps::set(F::sign - 1);
  auto pm = _mm512_and_si512(a, magmask), bm = _mm512_and_si512(b, magmask),
       cm = _mm512_and_si512(c, magmask);
  auto sign = _mm512_and_si512(_mm512_xor_si512(b, c), signbit);
  auto overflow = VectorOps::set(F::infinity - (Width == 16 && saturate));
  auto result = _mm512_or_si512(sign, VectorOps::select(VectorOps::gt(inf, pm), pm, overflow));
  if constexpr (Width != 16) {
    auto delta = VectorOps::sub(VectorOps::shr(cm, F::fraction), VectorOps::shr(bm, F::fraction));
    result = VectorOps::select(
        VectorOps::gt(VectorOps::set(uint64_t(-int64_t(F::bias + F::fraction))), delta), sign,
        result);
  }
  // Apply zero/infinity cases before NaN propagation; numerator NaNs take precedence over
  // denominator NaNs.
  auto bz = VectorOps::eq(bm, zero), cz = VectorOps::eq(cm, zero), bi = VectorOps::eq(bm, inf),
       ci = VectorOps::eq(cm, inf);
  result = VectorOps::select(VectorOps::either(cz, bi), sign, result);
  result = VectorOps::select(VectorOps::either(bz, ci), _mm512_or_si512(sign, inf), result);
  result = VectorOps::select(VectorOps::either(VectorOps::both(bz, cz), VectorOps::both(bi, ci)),
                             VectorOps::set(F::sign | F::infinity | F::quiet), result);
  const bool quiet_snan = !Rdna3 || ieee;
  const auto quiet = VectorOps::set(quiet_snan ? F::quiet : 0);
  result = VectorOps::select(VectorOps::gt(bm, inf), _mm512_or_si512(b, quiet), result);
  result = VectorOps::select(VectorOps::gt(cm, inf), _mm512_or_si512(c, quiet), result);

  // Output modifiers apply to the fixed-up result, not the provisional quotient.
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
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
    if constexpr (Rdna3) {
      if (dx10_clamp_off) {
        result = VectorOps::select(VectorOps::gt(VectorOps::band(unclamped, magmask), inf),
                                   unclamped, result);
      }
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
    if constexpr (Width == 16) {
      unsigned shift = mode & (RJ_ARITH_ALU_HIGH_A << operand) ? 16 : 0;
      result = _mm512_and_si512(_mm512_srl_epi32(result, _mm_cvtsi32_si128(int(shift))),
                                VectorOps::set(65535));
    }
  }
  auto keep = VectorOps::set(mode & (RJ_ARITH_ALU_ABS_A << operand) ? F::sign - 1 : ~uint64_t(0));
  auto flip = VectorOps::set(mode & (RJ_ARITH_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm512_xor_si512(_mm512_and_si512(result, keep), flip);
}

} // namespace

namespace rocjitsu::arith {

// Fix up division results and special values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3>
void fixup_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int lanes = Width == 64 ? 8 : 16;
  uint32_t staged[2][32];
  for (int lane = 0; lane < 32; lane += lanes) {
    auto result =
        value<Width, Rdna3>(load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1),
                            load<Width>(c, lane, mode, 2), mode, flags & RJ_ARITH_FP16_OVFL,
                            flags & RJ_ARITH_FP_DX10_CLAMP_OFF, flags & RJ_ARITH_FP_IEEE_MODE);
    if constexpr (Width == 64) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                          _mm512_cvtepi64_epi32(result));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
    } else {
      if constexpr (Width == 16) {
        unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d[0] + lane));
        result = _mm512_or_si512(_mm512_andnot_si512(_mm512_set1_epi32(int(65535u << shift)), old),
                                 _mm512_sll_epi32(result, _mm_cvtsi32_si128(int(shift))));
      }
      _mm512_mask_storeu_epi32(d[0] + lane, __mmask16(exec_mask >> lane), result);
    }
  }
  // Delay FP64 writes for cross-half aliases; D1 wins when D0 and D1 alias.
  if constexpr (Width == 64) {
    for (int reg = 0; reg < 2; ++reg) {
      for (int lane = 0; lane < 32; lane += 16) {
        auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
        _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane), result);
      }
    }
  }
}

template void fixup_x86_64_v4<16>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v4<32>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v4<64>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);

template void fixup_x86_64_v4<16, true>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                        const uint32_t *const *, const uint32_t *const *,
                                        const uint32_t *const *);
template void fixup_x86_64_v4<32, true>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                        const uint32_t *const *, const uint32_t *const *,
                                        const uint32_t *const *);
template void fixup_x86_64_v4<64, true>(uint32_t, uint32_t, uint64_t, uint32_t *const *,
                                        const uint32_t *const *, const uint32_t *const *,
                                        const uint32_t *const *);

} // namespace rocjitsu::arith

namespace rocjitsu::arith {

// Return the OR of active-lane exception causes, or zero under CLAMP.
// Never writes operands. Requires the named CPU features and validated controls.
template <int Width>
uint32_t fixup_exceptions_x86_64_v4(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  if (mode & RJ_ARITH_ALU_CLAMP) {
    return 0;
  }

  // Retain the cheap existing zero-cause test for ordinary FP32 waves.
  // A failed check now uses SIMD classification rather than scalar fallback.
  if constexpr (Width == 32) {
    if (!(mode & RJ_ARITH_ALU_OMOD_MASK)) {
      bool ordinary = true;
      const uint32_t *inputs[] = {a[0], b[0], c[0]};
      for (const auto *input : inputs) {
        for (int lane = 0; lane < 32; lane += 16) {
          auto exponent = _mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512(input + lane), 23),
                                           _mm512_set1_epi32(255));
          ordinary &= !_mm512_cmp_epu32_mask(_mm512_sub_epi32(exponent, _mm512_set1_epi32(96)),
                                             _mm512_set1_epi32(62), _MM_CMPINT_GT);
        }
      }
      if (ordinary) {
        return 0;
      }
    }
  }
  constexpr int lanes = Width == 64 ? 8 : 16;
  uint32_t causes = 0;
  for (int lane = 0; lane < 32; lane += lanes) {
    causes |= fixup_exception_lanes<Width, Ops<Width == 64>>(
        load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1), load<Width>(c, lane, mode, 2),
        mode, exec_mask >> lane);
  }
  return causes;
}

template uint32_t fixup_exceptions_x86_64_v4<16>(uint32_t, uint32_t, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *);
template uint32_t fixup_exceptions_x86_64_v4<32>(uint32_t, uint32_t, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *);
template uint32_t fixup_exceptions_x86_64_v4<64>(uint32_t, uint32_t, const uint32_t *const *,
                                                 const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith
