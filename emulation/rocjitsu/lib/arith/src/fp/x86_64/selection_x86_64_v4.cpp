// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP16/FP32/FP64 binary and FP16/FP32 ternary selection, including legacy NaN priorities.
// Integer encoding operations preserve selected signaling NaNs and implement
// OMOD without changing host rounding; staged masked stores permit VGPR aliasing.

#include "common/packed_alu.h"
#include "fp/binary.h"
#include "fp/float_output.h"
#include "fp/fp64.h"
#include "fp/half_binary.h"
#include "fp/minmax.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Broadcast an encoding at the selected element width.
template <int Width> __m512i splat(uint64_t value) {
  if constexpr (Width <= 32) {
    return _mm512_set1_epi32(int32_t(value));
  } else {
    return _mm512_set1_epi64(int64_t(value));
  }
}

// Select complete elements under a lane mask.
template <int Width> __m512i select(__m512i a, unsigned lanes, __m512i b) {
  if constexpr (Width <= 32) {
    return _mm512_mask_mov_epi32(a, __mmask16(lanes), b);
  } else {
    return _mm512_mask_mov_epi64(a, __mmask8(lanes), b);
  }
}

// Compare unsigned encodings, independently of host floating-point state.
template <int Width> unsigned less(__m512i a, __m512i b) {
  if constexpr (Width <= 32) {
    return _mm512_cmplt_epu32_mask(a, b);
  } else {
    return _mm512_cmplt_epu64_mask(a, b);
  }
}

// Compare encodings for equality.
template <int Width> unsigned equal(__m512i a, __m512i b) {
  if constexpr (Width <= 32) {
    return _mm512_cmpeq_epi32_mask(a, b);
  } else {
    return _mm512_cmpeq_epi64_mask(a, b);
  }
}

// Add an exponent adjustment without modifying the significand.
template <int Width> __m512i add(__m512i a, __m512i b) {
  if constexpr (Width <= 32) {
    return _mm512_add_epi32(a, b);
  } else {
    return _mm512_add_epi64(a, b);
  }
}

// Prepare one source's sign modifiers and guest denormal policy.
template <int Width> __m512i prepare(__m512i value, uint64_t flags, uint32_t mode) {
  using F = IeeeFormat<Width>;
  auto sign = splat<Width>(F::sign), magnitude = splat<Width>(F::sign - 1);
  if (mode & RJ_ARITH_ALU_ABS_A) {
    value = _mm512_and_si512(value, magnitude);
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    value = _mm512_xor_si512(value, sign);
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    auto tiny = less<Width>(_mm512_and_si512(value, magnitude), splat<Width>(1ULL << F::fraction));
    value = select<Width>(value, tiny, _mm512_and_si512(value, sign));
  }
  return value;
}

// Select between prepared encodings without applying output modifiers.
template <int Width>
__m512i pick(__m512i a, __m512i b, uint64_t flags, bool maximum, bool legacy, bool first_nan,
             bool propagate) {
  using F = IeeeFormat<Width>;
  constexpr auto sign = F::sign;
  auto zero = _mm512_setzero_si512();
  auto sign_bits = splat<Width>(sign), magnitude = splat<Width>(sign - 1);
  auto inf = splat<Width>(F::infinity), q = splat<Width>(F::quiet);
  auto an = less<Width>(inf, _mm512_and_si512(a, magnitude));
  auto bn = less<Width>(inf, _mm512_and_si512(b, magnitude));
  // Unsigned sortable keys order finite encodings numerically, including -0 < +0.
  auto key = [&](auto v) {
    auto negative = equal<Width>(_mm512_and_si512(v, sign_bits), sign_bits);
    return _mm512_xor_si512(v, select<Width>(sign_bits, negative, splat<Width>(sign | (sign - 1))));
  };
  auto choose_b = maximum ? less<Width>(key(a), key(b)) : less<Width>(key(b), key(a));
  auto value = select<Width>(a, choose_b, b);
  value = select<Width>(value, an, b);
  value = select<Width>(value, bn, a);
  auto chosen_nan = first_nan ? a : b;
  value = select<Width>(value, an & bn, legacy ? chosen_nan : _mm512_or_si512(a, q));
  if (propagate) {
    auto first = select<Width>(b, an, a);
    value = select<Width>(value, an | bn, _mm512_or_si512(first, q));
  }
  if (legacy && (flags & RJ_ARITH_FP_IEEE_MODE)) {
    auto as = an & equal<Width>(_mm512_and_si512(a, q), zero);
    auto bs = bn & equal<Width>(_mm512_and_si512(b, q), zero);
    value = select<Width>(value, bs, _mm512_or_si512(b, q));
    value = select<Width>(value, as, _mm512_or_si512(a, q));
  }
  return value;
}

// Apply OMOD and CLAMP once to the final selected encoding.
template <int Width> __m512i finish(__m512i value, uint64_t flags, uint32_t mode) {
  using F = IeeeFormat<Width>;
  constexpr auto fraction = F::fraction;
  constexpr auto infinity = F::infinity;
  auto zero = _mm512_setzero_si512();
  auto sign_bits = splat<Width>(F::sign), magnitude = splat<Width>(F::sign - 1);
  auto inf = splat<Width>(infinity);
  const unsigned omod = (mode >> 6) & 3;
  if (omod) {
    auto mag = _mm512_and_si512(value, magnitude);
    // Guest OMOD flushes tiny inputs to +0; halving a minimum-exponent normal
    // produces signed zero. Other normal powers-of-two scaling is exact.
    auto tiny = less<Width>(mag, splat<Width>(1ULL << fraction));
    value = select<Width>(value, tiny, zero);
    mag = _mm512_and_si512(value, magnitude);
    auto normal = less<Width>(zero, mag) & less<Width>(mag, inf);
    auto delta = splat<Width>(omod == 3 ? -(1ULL << fraction) : uint64_t(omod) << fraction);
    auto scaled = add<Width>(value, delta);
    if (omod == 3) {
      auto underflow = less<Width>(mag, splat<Width>(2ULL << fraction));
      scaled = select<Width>(scaled, underflow, _mm512_and_si512(value, sign_bits));
    } else {
      auto overflow = ~less<Width>(mag, splat<Width>(infinity - (uint64_t(omod) << fraction)));
      unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
      auto negative = equal<Width>(_mm512_and_si512(value, sign_bits), sign_bits);
      auto to_inf = rounding == 0 ? ~0U : rounding == 1 ? ~negative : rounding == 2 ? negative : 0;
      if constexpr (Width == 16) {
        // FP16_OVFL saturates finite overflow even when rounding would select infinity.
        if (flags & RJ_ARITH_FP16_OVFL) {
          to_inf = 0;
        }
      }
      auto limit = select<Width>(splat<Width>(infinity - 1), to_inf, inf);
      limit = _mm512_or_si512(limit, _mm512_and_si512(value, sign_bits));
      scaled = select<Width>(scaled, overflow, limit);
    }
    value = select<Width>(value, normal, scaled);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    auto mag = _mm512_and_si512(value, magnitude);
    auto nan = less<Width>(inf, mag);
    auto negative = equal<Width>(_mm512_and_si512(value, sign_bits), sign_bits);
    auto one = splat<Width>(Width == 16   ? 0x3c00ULL
                            : Width == 32 ? 0x3f800000ULL
                                          : 0x3ff0000000000000ULL);
    auto clamped = select<Width>(value, less<Width>(one, mag), one);
    clamped = select<Width>(clamped, negative | nan, zero);
    value = select<Width>(clamped, flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? nan : 0, value);
  }
  return value;
}

// Evaluate a binary instruction with prepared sources and one output-modifier step.
template <int Width>
__m512i evaluate(__m512i a, __m512i b, uint64_t flags, uint32_t mode, bool maximum, bool legacy,
                 bool first_nan, bool propagate) {
  auto value = pick<Width>(prepare<Width>(a, flags, mode), prepare<Width>(b, flags, mode >> 1),
                           flags, maximum, legacy, first_nan, propagate);
  return finish<Width>(value, flags, mode);
}

// Execute a binary selection after validation, preserving inactive lanes and overlapping VGPRs.
template <int Width, class Op>
void run(Op op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b) {
  bool maximum =
      op == Op::MaxNum || op == Op::Maximum || op == Op::MaxLegacy || op == Op::MaxLegacyCdna;
  bool legacy = op == Op::MinLegacy || op == Op::MaxLegacy || op == Op::MinLegacyCdna ||
                op == Op::MaxLegacyCdna;
  bool first_nan = op != Op::MinLegacyCdna && op != Op::MaxLegacyCdna;
  bool propagate = op == Op::Minimum || op == Op::Maximum;
  uint32_t result[Width / 32][32];
  for (int lane = 0; lane < 32; lane += 512 / Width) {
    auto load = [&](const uint32_t *const *v) {
      if constexpr (Width <= 32) {
        return _mm512_loadu_si512(v[0] + lane);
      } else {
        auto lo = _mm512_cvtepu32_epi64(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[0] + lane)));
        auto hi = _mm512_cvtepu32_epi64(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[1] + lane)));
        return _mm512_or_si512(lo, _mm512_slli_epi64(hi, 32));
      }
    };
    auto value =
        evaluate<Width>(load(a), load(b), flags, mode, maximum, legacy, first_nan, propagate);
    if constexpr (Width <= 32) {
      _mm512_storeu_si512(result[0] + lane, value);
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                          _mm512_cvtepi64_epi32(value));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(value, 32)));
    }
  }
  for (int reg = 0; reg < Width / 32; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

// Execute a ternary selection, staging all source reads before masked destination writes.
template <int Width, Minmax3 Op>
void ternary_run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                 const uint32_t *b, const uint32_t *c) {
  using P = MinmaxPolicy<Op>;
  auto select_pair = [&](auto x, auto y, bool maximum) {
    return pick<Width>(x, y, flags, maximum, P::legacy, !P::cdna, P::propagate_nan);
  };
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += 16) {
    auto load = [&](const uint32_t *v, int operand) {
      auto bits = _mm512_loadu_si512(v + lane);
      if constexpr (Width == 16) {
        if (mode & (RJ_ARITH_ALU_HIGH_A << operand)) {
          bits = _mm512_srli_epi32(bits, 16);
        }
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(0xffff));
      }
      return prepare<Width>(bits, flags, mode >> operand);
    };
    auto x = load(a, 0), y = load(b, 1), z = load(c, 2);
    __m512i value;
    if constexpr (P::median) {
      if constexpr (P::cdna) {
        auto ab = select_pair(x, y, true);
        auto largest = select_pair(ab, z, true);
        auto numeric_equal = [&](auto v) {
          auto magnitude = _mm512_and_si512(_mm512_or_si512(largest, v),
                                            splat<Width>(IeeeFormat<Width>::sign - 1));
          return equal<Width>(largest, v) | equal<Width>(magnitude, _mm512_setzero_si512());
        };
        value = select<Width>(ab, numeric_equal(y), select_pair(x, z, true));
        value = select<Width>(value, numeric_equal(x), select_pair(y, z, true));
      } else {
        value = select_pair(select_pair(x, y, false),
                            select_pair(select_pair(x, y, true), z, false), true);
      }
      auto is_nan = [&](auto v) {
        return less<Width>(splat<Width>(IeeeFormat<Width>::infinity),
                           _mm512_and_si512(v, splat<Width>(IeeeFormat<Width>::sign - 1)));
      };
      auto minimum = select_pair(select_pair(x, y, false), z, false);
      value = select<Width>(value, is_nan(x) | is_nan(y) | is_nan(z), minimum);
    } else {
      value = select_pair(select_pair(x, y, P::first_maximum), z, P::second_maximum);
    }
    value = finish<Width>(value, flags, mode);
    if constexpr (Width == 16) {
      bool high = mode & RJ_ARITH_ALU_HIGH_D;
      auto previous = _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                       _mm512_set1_epi32(high ? 0xffff : int(0xffff0000U)));
      value = _mm512_or_si512(previous, high ? _mm512_slli_epi32(value, 16) : value);
    }
    _mm512_storeu_si512(result + lane, value);
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
}

// Select a compile-time operation policy outside the SIMD loop.
template <int Width>
void ternary_dispatch(Minmax3 op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                      const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  switch (op) {
  case Minmax3::Min3Cdna:
    return ternary_run<Width, Minmax3::Min3Cdna>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Cdna:
    return ternary_run<Width, Minmax3::Max3Cdna>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianCdna:
    return ternary_run<Width, Minmax3::MedianCdna>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Min3Legacy:
    return ternary_run<Width, Minmax3::Min3Legacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Legacy:
    return ternary_run<Width, Minmax3::Max3Legacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxLegacy:
    return ternary_run<Width, Minmax3::MinmaxLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminLegacy:
    return ternary_run<Width, Minmax3::MaxminLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianLegacy:
    return ternary_run<Width, Minmax3::MedianLegacy>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MedianNum:
    return ternary_run<Width, Minmax3::MedianNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Min3Num:
    return ternary_run<Width, Minmax3::Min3Num>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Max3Num:
    return ternary_run<Width, Minmax3::Max3Num>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinmaxNum:
    return ternary_run<Width, Minmax3::MinmaxNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaxminNum:
    return ternary_run<Width, Minmax3::MaxminNum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Minimum3:
    return ternary_run<Width, Minmax3::Minimum3>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::Maximum3:
    return ternary_run<Width, Minmax3::Maximum3>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MinimumMaximum:
    return ternary_run<Width, Minmax3::MinimumMaximum>(flags, exec_mask, mode, d, a, b, c);
  case Minmax3::MaximumMinimum:
    return ternary_run<Width, Minmax3::MaximumMinimum>(flags, exec_mask, mode, d, a, b, c);
  }
}

} // namespace

// Evaluate an already-validated FP32 selection using AVX-512 lane masks.
void selection_f32_x86_64_v4(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b) {
  run<32>(op, flags, exec_mask, mode, d, a, b);
}

// Evaluate an already-validated FP64 selection using AVX-512 lane masks.
void selection_f64_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b) {
  run<64>(op, flags, exec_mask, mode, d, a, b);
}

// Select ordinary or packed FP16 operands without FP16 conversion side effects.
// Unselected destination halves and inactive lanes retain their previous contents.
void selection_f16_x86_64_v4(Binary op, bool packed, uint64_t flags, uint32_t exec_mask,
                             uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b) {
  const bool maximum = op == Binary::MaxNum || op == Binary::Maximum || op == Binary::MaxLegacy ||
                       op == Binary::MaxLegacyCdna;
  const bool legacy = op == Binary::MinLegacy || op == Binary::MaxLegacy ||
                      op == Binary::MinLegacyCdna || op == Binary::MaxLegacyCdna;
  const bool first_nan = op != Binary::MinLegacyCdna && op != Binary::MaxLegacyCdna;
  const bool propagate = op == Binary::Minimum || op == Binary::Maximum;
  const uint32_t modes[] = {packed ? packed_half_instruction_flags(mode, false) : mode,
                            packed_half_instruction_flags(mode, true)};
  const int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += 16) {
    const auto av = _mm512_loadu_si512(a + lane), bv = _mm512_loadu_si512(b + lane);
    auto value = _mm512_setzero_si512();
    for (int half = 0; half < (packed ? 2 : 1); ++half) {
      const uint32_t m = modes[half];
      auto x = m & RJ_ARITH_ALU_HIGH_A ? _mm512_srli_epi32(av, 16) : av;
      auto y = m & RJ_ARITH_ALU_HIGH_B ? _mm512_srli_epi32(bv, 16) : bv;
      x = _mm512_and_si512(x, _mm512_set1_epi32(0xffff));
      y = _mm512_and_si512(y, _mm512_set1_epi32(0xffff));
      auto selected = evaluate<16>(x, y, flags, m, maximum, legacy, first_nan, propagate);
      value = _mm512_or_si512(value, half ? _mm512_slli_epi32(selected, 16) : selected);
    }
    if (!packed) {
      auto previous = _mm512_loadu_si512(d + lane);
      auto keep = _mm512_set1_epi32(destination_shift ? 0xffff : int(0xffff0000U));
      value = _mm512_or_si512(_mm512_and_si512(previous, keep),
                              destination_shift ? _mm512_slli_epi32(value, 16) : value);
    }
    _mm512_storeu_si512(result + lane, value);
  }
  for (int lane = 0; lane < 32; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane),
                             _mm512_loadu_si512(result + lane));
  }
}

// Evaluate a validated FP16 or FP32 ternary selection with AVX-512.
void minmax3_x86_64_v4(Minmax3 op, bool half, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                       uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (half) {
    ternary_dispatch<16>(op, flags, exec_mask, mode, d, a, b, c);
  } else {
    ternary_dispatch<32>(op, flags, exec_mask, mode, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
