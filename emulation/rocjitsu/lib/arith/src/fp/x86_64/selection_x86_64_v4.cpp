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

// Map IEEE encodings to signed integer keys. Applying the transform twice restores the bits.
// Left-align half encodings first so all 32-bit comparisons use the same sign position.
template <int Width> __m512i order_key(__m512i bits) {
  if constexpr (Width == 64) {
    return _mm512_xor_si512(bits, _mm512_srli_epi64(_mm512_srai_epi64(bits, 63), 1));
  } else {
    return _mm512_xor_si512(bits, _mm512_srli_epi32(_mm512_srai_epi32(bits, 31), 1));
  }
}

template <int Width> __m512i to_key(__m512i bits) {
  if constexpr (Width == 16) {
    bits = _mm512_slli_epi32(bits, 16);
  }
  return order_key<Width>(bits);
}

template <int Width> __m512i from_key(__m512i key) {
  auto bits = order_key<Width>(key);
  if constexpr (Width == 16) {
    return _mm512_srli_epi32(bits, 16);
  }
  return bits;
}

template <int Width> __m512i ordered_pair(__m512i a, __m512i b, bool maximum) {
  if constexpr (Width == 64) {
    return maximum ? _mm512_max_epi64(a, b) : _mm512_min_epi64(a, b);
  } else {
    return maximum ? _mm512_max_epi32(a, b) : _mm512_min_epi32(a, b);
  }
}

// Select between prepared encodings without applying output modifiers.
template <int Width>
__m512i pick(__m512i a, __m512i b, uint64_t flags, bool maximum, bool legacy, bool first_nan,
             bool propagate) {
  using F = IeeeFormat<Width>;
  constexpr auto sign = F::sign;
  auto zero = _mm512_setzero_si512();
  auto magnitude = splat<Width>(sign - 1);
  auto inf = splat<Width>(F::infinity), q = splat<Width>(F::quiet);
  auto an = less<Width>(inf, _mm512_and_si512(a, magnitude));
  auto bn = less<Width>(inf, _mm512_and_si512(b, magnitude));
  auto ka = to_key<Width>(a), kb = to_key<Width>(b);
  __m512i value;
  if constexpr (Width == 32) {
    auto take_b = maximum ? _mm512_cmplt_epi32_mask(ka, kb) : _mm512_cmplt_epi32_mask(kb, ka);
    value = _mm512_mask_blend_epi32(take_b, a, b);
  } else {
    value = from_key<Width>(ordered_pair<Width>(ka, kb, maximum));
  }
  if (!(an | bn)) {
    return value;
  }
  if (propagate) {
    auto first = select<Width>(b, an, a);
    return select<Width>(value, an | bn, _mm512_or_si512(first, q));
  }
  value = select<Width>(value, an, b);
  value = select<Width>(value, bn, a);
  auto chosen_nan = first_nan ? a : b;
  value = select<Width>(value, an & bn, legacy ? chosen_nan : _mm512_or_si512(a, q));
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
    __m512i clamped;
    if constexpr (Width == 16) {
      clamped = select<Width>(value, less<Width>(one, mag), one);
      clamped = select<Width>(clamped, negative, zero);
    } else {
      clamped = ordered_pair<Width>(zero, ordered_pair<Width>(one, value, false), true);
    }
    value = select<Width>(clamped, nan, flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? value : zero);
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
template <int Width, auto op>
[[gnu::noinline]] void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  using Op = decltype(op);
  bool maximum =
      op == Op::MaxNum || op == Op::Maximum || op == Op::MaxLegacy || op == Op::MaxLegacyCdna;
  bool legacy = op == Op::MinLegacy || op == Op::MaxLegacy || op == Op::MinLegacyCdna ||
                op == Op::MaxLegacyCdna;
  bool first_nan = op != Op::MinLegacyCdna && op != Op::MaxLegacyCdna;
  bool propagate = op == Op::Minimum || op == Op::Maximum;
  if constexpr (Width == 32) {
    if ((mode &
         (RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B)) &&
        !(mode & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP))) {
      // MIN/MAX with suppressed exceptions selects normal values directly. Handle
      // zeros, denormals and NaNs by encoding to preserve guest ties and payloads.
      __m512i result[2];
      for (int part = 0; part < 2; ++part) {
        auto x = prepare<32>(_mm512_loadu_si512(a[0] + 16 * part), flags, mode);
        auto y = prepare<32>(_mm512_loadu_si512(b[0] + 16 * part), flags, mode >> 1);
        auto xf = _mm512_castsi512_ps(x), yf = _mm512_castsi512_ps(y);
        auto special = _mm512_fpclass_ps_mask(xf, 0xa7) | _mm512_fpclass_ps_mask(yf, 0xa7);
        if (__builtin_expect(special != 0, 0)) {
          result[part] = pick<32>(x, y, flags, maximum, legacy, first_nan, propagate);
        } else {
          auto selected = maximum ? _mm512_max_round_ps(xf, yf, _MM_FROUND_NO_EXC)
                                  : _mm512_min_round_ps(xf, yf, _MM_FROUND_NO_EXC);
          result[part] = _mm512_castps_si512(selected);
        }
      }
      for (int part = 0; part < 2; ++part) {
        _mm512_mask_storeu_epi32(d[0] + 16 * part, __mmask16(exec_mask >> (16 * part)),
                                 result[part]);
      }
      return;
    }
  }
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
[[gnu::noinline]] void ternary_run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                   const uint32_t *a, const uint32_t *b, const uint32_t *c) {
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
    using F = IeeeFormat<Width>;
    auto nan = [&](auto v) {
      return less<Width>(splat<Width>(F::infinity), _mm512_and_si512(v, splat<Width>(F::sign - 1)));
    };
    auto nx = nan(x), ny = nan(y), nz = nan(z);
    bool sensitive = false;
    if constexpr (P::legacy) {
      if (flags & RJ_ARITH_FP_IEEE_MODE) {
        auto signaling = [&](auto v, unsigned n) {
          return n &
                 equal<Width>(_mm512_and_si512(v, splat<Width>(F::quiet)), _mm512_setzero_si512());
        };
        sensitive = signaling(x, nx) | signaling(y, ny) | signaling(z, nz);
      }
      if constexpr (P::cdna && P::median) {
        auto zero = [&](auto v) {
          return equal<Width>(_mm512_and_si512(v, splat<Width>(F::sign - 1)),
                              _mm512_setzero_si512());
        };
        sensitive |= bool(zero(x) | zero(y) | zero(z));
      }
    }
    if (!sensitive) {
      auto kx = to_key<Width>(x), ky = to_key<Width>(y), kz = to_key<Width>(z);
      auto first = ordered_pair<Width>(kx, ky, P::first_maximum);
      auto key = ordered_pair<Width>(first, kz, P::second_maximum);
      if constexpr (P::median) {
        auto lo = _mm512_min_epi32(kx, ky), hi = _mm512_max_epi32(kx, ky);
        key = _mm512_max_epi32(lo, _mm512_min_epi32(hi, kz));
      }
      // A numeric selector skips NaNs; all-NaN inputs retain their policy-specific payload.
      auto any = nx | ny | nz;
      if constexpr (P::propagate_nan) {
        auto chosen = select<Width>(select<Width>(z, ny, y), nx, x);
        value = select<Width>(from_key<Width>(key), any,
                              _mm512_or_si512(chosen, splat<Width>(F::quiet)));
      } else if (any) {
        auto skip = [&](auto k, unsigned n, bool maximum) {
          return select<Width>(k, n, _mm512_set1_epi32(maximum ? INT32_MIN : INT32_MAX));
        };
        __m512i ordered;
        if constexpr (P::median) {
          ordered = _mm512_min_epi32(skip(kx, nx, false),
                                     _mm512_min_epi32(skip(ky, ny, false), skip(kz, nz, false)));
        } else {
          auto pair = ordered_pair<Width>(skip(kx, nx, P::first_maximum),
                                          skip(ky, ny, P::first_maximum), P::first_maximum);
          ordered = ordered_pair<Width>(skip(pair, nx & ny, P::second_maximum),
                                        skip(kz, nz, P::second_maximum), P::second_maximum);
        }
        auto all = P::legacy ? (P::cdna ? z : x) : _mm512_or_si512(x, splat<Width>(F::quiet));
        value = select<Width>(from_key<Width>(select<Width>(key, any, ordered)), nx & ny & nz, all);
      } else {
        value = from_key<Width>(key);
      }
    } else {

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
        auto minimum = select_pair(select_pair(x, y, false), z, false);
        value = select<Width>(value, nx | ny | nz, minimum);
      } else {
        value = select_pair(select_pair(x, y, P::first_maximum), z, P::second_maximum);
      }
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
  switch (op) {
  case Binary::MinNum:
    return run<32, Binary::MinNum>(flags, exec_mask, mode, d, a, b);
  case Binary::MaxNum:
    return run<32, Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
  case Binary::Minimum:
    return run<32, Binary::Minimum>(flags, exec_mask, mode, d, a, b);
  case Binary::Maximum:
    return run<32, Binary::Maximum>(flags, exec_mask, mode, d, a, b);
  case Binary::MinLegacy:
    return run<32, Binary::MinLegacy>(flags, exec_mask, mode, d, a, b);
  case Binary::MinLegacyCdna:
    return run<32, Binary::MinLegacyCdna>(flags, exec_mask, mode, d, a, b);
  case Binary::MaxLegacyCdna:
    return run<32, Binary::MaxLegacyCdna>(flags, exec_mask, mode, d, a, b);
  case Binary::MaxLegacy:
    return run<32, Binary::MaxLegacy>(flags, exec_mask, mode, d, a, b);
  default:
    __builtin_unreachable();
  }
}

namespace {
// Pack the two selected source halves and their independent sign modifiers.
__m512i prepare_packed(__m512i v, uint32_t lo, uint32_t hi) {
  auto even = lo & RJ_ARITH_ALU_HIGH_A ? _mm512_srli_epi32(v, 16) : v;
  auto odd = hi & RJ_ARITH_ALU_HIGH_A ? v : _mm512_slli_epi32(v, 16);
  int sign = (lo & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0) | (hi & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  return _mm512_xor_si512(_mm512_mask_blend_epi16(0xaaaaaaaau, even, odd), _mm512_set1_epi32(sign));
}

// Select FP16 or BF16 encodings directly. Signed integer keys distinguish opposite zeros.
template <bool Bf16>
__m512i pick_halfword(__m512i x, __m512i y, Binary op, uint64_t flags, bool clamp) {
  constexpr int inf = Bf16 ? 0x7f80 : 0x7c00, quiet = Bf16 ? 0x40 : 0x200;
  auto zero = _mm512_setzero_si512(), mag = _mm512_set1_epi16(0x7fff);
  auto mx = _mm512_and_si512(x, mag), my = _mm512_and_si512(y, mag);
  if (!Bf16 && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
    x = _mm512_mask_mov_epi16(x, _mm512_cmplt_epu16_mask(mx, _mm512_set1_epi16(0x400)),
                              _mm512_andnot_si512(mag, x));
    y = _mm512_mask_mov_epi16(y, _mm512_cmplt_epu16_mask(my, _mm512_set1_epi16(0x400)),
                              _mm512_andnot_si512(mag, y));
  }
  auto nx = _mm512_cmpgt_epu16_mask(mx, _mm512_set1_epi16(inf));
  auto ny = _mm512_cmpgt_epu16_mask(my, _mm512_set1_epi16(inf));
  auto key = [](__m512i v) {
    return _mm512_xor_si512(v, _mm512_srli_epi16(_mm512_srai_epi16(v, 15), 1));
  };
  bool maximum = op == Binary::MaxNum || op == Binary::Maximum || op == Binary::MaxLegacy ||
                 op == Binary::MaxLegacyCdna;
  auto kx = key(x), ky = key(y);
  auto chosen = _mm512_mask_blend_epi16(
      maximum ? _mm512_cmplt_epi16_mask(kx, ky) : _mm512_cmplt_epi16_mask(ky, kx), x, y);
  if (nx | ny) {
    bool legacy = op == Binary::MinLegacy || op == Binary::MaxLegacy ||
                  op == Binary::MinLegacyCdna || op == Binary::MaxLegacyCdna;
    auto q = _mm512_set1_epi16(quiet);
    auto first = _mm512_mask_blend_epi16(nx, y, x);
    if (op == Binary::Maximum || op == Binary::Minimum) {
      chosen = _mm512_mask_mov_epi16(chosen, nx | ny, _mm512_or_si512(first, q));
    } else {
      chosen = _mm512_mask_mov_epi16(_mm512_mask_mov_epi16(chosen, nx, y), ny, x);
      auto both = legacy ? (op == Binary::MinLegacyCdna || op == Binary::MaxLegacyCdna ? y : x)
                         : _mm512_or_si512(x, q);
      chosen = _mm512_mask_mov_epi16(chosen, nx & ny, both);
      if (legacy && (flags & RJ_ARITH_FP_IEEE_MODE)) {
        auto sx = nx & _mm512_cmpeq_epi16_mask(_mm512_and_si512(x, q), zero);
        auto sy = ny & _mm512_cmpeq_epi16_mask(_mm512_and_si512(y, q), zero);
        chosen = _mm512_mask_mov_epi16(chosen, sy, _mm512_or_si512(y, q));
        chosen = _mm512_mask_mov_epi16(chosen, sx, _mm512_or_si512(x, q));
      }
    }
  }
  if (clamp) {
    auto n = _mm512_cmpgt_epu16_mask(_mm512_and_si512(chosen, mag), _mm512_set1_epi16(inf));
    auto bounded =
        _mm512_max_epi16(zero, _mm512_min_epi16(chosen, _mm512_set1_epi16(Bf16 ? 0x3f80 : 0x3c00)));
    chosen = _mm512_mask_mov_epi16(bounded, n, flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? chosen : zero);
  }
  return chosen;
}

template <bool Bf16>
void packed_selection(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                      const uint32_t *a, const uint32_t *b) {
  auto lo = packed_half_instruction_flags(mode, false),
       hi = packed_half_instruction_flags(mode, true);
  __m512i results[2];
  for (int part = 0; part < 2; ++part) {
    auto x = prepare_packed(_mm512_loadu_si512(a + 16 * part), lo, hi);
    auto y = prepare_packed(_mm512_loadu_si512(b + 16 * part), lo >> 1, hi >> 1);
    results[part] = pick_halfword<Bf16>(x, y, op, flags, mode & RJ_ARITH_PK_CLAMP);
  }
  for (int part = 0; part < 2; ++part) {
    _mm512_mask_storeu_epi32(d + 16 * part, __mmask16(exec_mask >> (16 * part)), results[part]);
  }
}
} // namespace
// The packed BF16 API uses numeric min/max independently of guest FP mode.
void packed_bf16_selection_x86_64_v4(bool maximum, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                     const uint32_t *a, const uint32_t *b) {
  packed_selection<true>(maximum ? Binary::MaxNum : Binary::MinNum, 0, exec_mask, mode, d, a, b);
}

// Evaluate an already-validated FP64 selection using AVX-512 lane masks.
void selection_f64_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b) {
  switch (op) {
  case Fp64::MinNum:
    return run<64, Fp64::MinNum>(flags, exec_mask, mode, d, a, b);
  case Fp64::MaxNum:
    return run<64, Fp64::MaxNum>(flags, exec_mask, mode, d, a, b);
  case Fp64::Minimum:
    return run<64, Fp64::Minimum>(flags, exec_mask, mode, d, a, b);
  case Fp64::Maximum:
    return run<64, Fp64::Maximum>(flags, exec_mask, mode, d, a, b);
  case Fp64::MinLegacy:
    return run<64, Fp64::MinLegacy>(flags, exec_mask, mode, d, a, b);
  case Fp64::MinLegacyCdna:
    return run<64, Fp64::MinLegacyCdna>(flags, exec_mask, mode, d, a, b);
  case Fp64::MaxLegacyCdna:
    return run<64, Fp64::MaxLegacyCdna>(flags, exec_mask, mode, d, a, b);
  case Fp64::MaxLegacy:
    return run<64, Fp64::MaxLegacy>(flags, exec_mask, mode, d, a, b);
  default:
    __builtin_unreachable();
  }
}

// Select ordinary or packed FP16 operands without FP16 conversion side effects.
// Unselected destination halves and inactive lanes retain their previous contents.
void selection_f16_x86_64_v4(Binary op, bool packed, uint64_t flags, uint32_t exec_mask,
                             uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b) {
  if (packed) {
    return packed_selection<false>(op, flags, exec_mask, mode, d, a, b);
  }
  if (!(mode & RJ_ARITH_ALU_OMOD_MASK)) {
    auto source = [](const uint32_t *v, uint32_t m) {
      auto low = _mm512_loadu_si512(v), high = _mm512_loadu_si512(v + 16);
      if (m & RJ_ARITH_ALU_HIGH_A) {
        low = _mm512_srli_epi32(low, 16);
        high = _mm512_srli_epi32(high, 16);
      }
      auto bits = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(low)),
                                     _mm512_cvtepi32_epi16(high), 1);
      if (m & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi16(0x7fff));
      }
      return m & RJ_ARITH_ALU_NEG_A ? _mm512_xor_si512(bits, _mm512_set1_epi16(short(0x8000)))
                                    : bits;
    };
    auto bits = pick_halfword<false>(source(a, mode), source(b, mode >> 1), op, flags,
                                     mode & RJ_ARITH_ALU_CLAMP);
    auto low = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(bits));
    auto high = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(bits, 1));
    for (int part = 0; part < 2; ++part) {
      auto v = part ? high : low;
      auto old = _mm512_loadu_si512(d + 16 * part);
      auto out = mode & RJ_ARITH_ALU_HIGH_D
                     ? _mm512_mask_blend_epi16(0xaaaaaaaau, old, _mm512_slli_epi32(v, 16))
                     : _mm512_mask_blend_epi16(0xaaaaaaaau, v, old);
      _mm512_mask_storeu_epi32(d + 16 * part, __mmask16(exec_mask >> (16 * part)), out);
    }
    return;
  }
  const bool maximum = op == Binary::MaxNum || op == Binary::Maximum || op == Binary::MaxLegacy ||
                       op == Binary::MaxLegacyCdna;
  const bool legacy = op == Binary::MinLegacy || op == Binary::MaxLegacy ||
                      op == Binary::MinLegacyCdna || op == Binary::MaxLegacyCdna;
  const bool first_nan = op != Binary::MinLegacyCdna && op != Binary::MaxLegacyCdna;
  const bool propagate = op == Binary::Minimum || op == Binary::Maximum;
  const bool high = mode & RJ_ARITH_ALU_HIGH_D;
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(a + lane), y = _mm512_loadu_si512(b + lane);
    if (mode & RJ_ARITH_ALU_HIGH_A) {
      x = _mm512_srli_epi32(x, 16);
    }
    if (mode & RJ_ARITH_ALU_HIGH_B) {
      y = _mm512_srli_epi32(y, 16);
    }
    auto mask = _mm512_set1_epi32(0xffff);
    auto value = evaluate<16>(_mm512_and_si512(x, mask), _mm512_and_si512(y, mask), flags, mode,
                              maximum, legacy, first_nan, propagate);
    auto previous = _mm512_loadu_si512(d + lane);
    value = high ? _mm512_mask_blend_epi16(0xaaaaaaaau, previous, _mm512_slli_epi32(value, 16))
                 : _mm512_mask_blend_epi16(0xaaaaaaaau, value, previous);
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
