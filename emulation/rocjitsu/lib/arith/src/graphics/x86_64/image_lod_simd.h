// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Quantize four independent quad footprints with AVX2. Integer
// norms and the eight-segment logarithm retain the inherited scalar precision.
#pragma once
#include "common/bits.h"
#include "common/compiler.h"
#include "graphics/image_lod.h"
#include "graphics/image_lod_scalar.h"
#include "rj_arith/detail/image_footprint.h"
#include <immintrin.h>

// Helper definitions are isolated in the AVX2 compilation namespace.
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod_simd {
using D = __m256d;
using I = __m128i;
using Q = __m256i;
// Reinterpret the double vectors as raw 64-bit lanes.
inline Q bits(D value) { return _mm256_castpd_si256(value); }
// Reinterpret raw 64-bit lanes as doubles.
inline D from_bits(Q value) { return _mm256_castsi256_pd(value); }
// Create zero integer lanes at the footprint batch width.
inline I zero() { return _mm_setzero_si128(); }
// Load aligned intermediate integer lanes.
inline I load(const int32_t *value) { return _mm_load_si128(reinterpret_cast<const I *>(value)); }
// Store aligned intermediate integer lanes.
inline void store(int32_t *destination, I value) {
  _mm_store_si128(reinterpret_cast<I *>(destination), value);
}
// Compare doubles, returning full-bit vector masks.
template <int Predicate> inline D compare(D a, D b) { return _mm256_cmp_pd(a, b, Predicate); }
// Select B on full-bit mask lanes and A otherwise.
inline D select(D a, D b, D mask) {
  return _mm256_or_pd(_mm256_andnot_pd(mask, a), _mm256_and_pd(mask, b));
}
// Extract low DWORDs from four 64-bit vector elements.
inline I low_words(D value) {
  return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
      _mm256_castpd_si256(value), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));
}
// Construct exact normal powers of two; all supplied exponents are in [-1022,1023].
inline D power_two(I exponent) {
  return from_bits(
      _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm_add_epi32(exponent, _mm_set1_epi32(1023))), 52));
}
// Quantization uses floor, without modifying host rounding controls.
inline D floor(D value) {
  return _mm256_round_pd(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}
// Absolute magnitude of double lanes.
inline D magnitude(D value) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), value); }
// Extract frexp exponents of nonnegative doubles; zero returns -1022.
inline I exponents(D value) {
  return _mm_sub_epi32(low_words(from_bits(_mm256_srli_epi64(bits(value), 52))),
                       _mm_set1_epi32(1022));
}
// Extract one bit per double mask lane for exceptional-quad repair.
inline unsigned mask_bits(D mask) { return unsigned(_mm256_movemask_pd(mask)); }
// Fixed-point five-projection norm of signed integer components.
inline I norm(I x, I y) {
  const auto a = _mm_max_epi32(_mm_abs_epi32(x), _mm_abs_epi32(y));
  const auto b = _mm_min_epi32(_mm_abs_epi32(x), _mm_abs_epi32(y));
  const auto projection = [&](int u, int v) {
    return _mm_add_epi32(_mm_mullo_epi32(a, _mm_set1_epi32(u)),
                         _mm_mullo_epi32(b, _mm_set1_epi32(v)));
  };
  auto value = _mm_max_epi32(projection(256, 16), projection(250, 56));
  value = _mm_max_epi32(value, projection(238, 96));
  value = _mm_max_epi32(value, projection(221, 130));
  value = _mm_max_epi32(value, projection(197, 164));
  return _mm_srli_epi32(_mm_add_epi32(value, _mm_set1_epi32(128)), 8);
}
// Convert bounded integer norms to fixed8 LOD; every norm fits exactly in FP32.
inline I logarithm(I value, I exponent, int zero_lod) {
  const auto positive = _mm_max_epi32(value, _mm_set1_epi32(1));
  const auto bits = _mm_castps_si128(_mm_cvtepi32_ps(positive));
  const auto leading = _mm_sub_epi32(_mm_srli_epi32(bits, 23), _mm_set1_epi32(127));
  const auto denominator = _mm_sllv_epi32(_mm_set1_epi32(1), leading);
  const auto fraction = _mm_slli_epi32(_mm_sub_epi32(positive, denominator), 10);
  const auto shift = _mm_add_epi32(leading, _mm_set1_epi32(7));
  const auto segment = _mm_srlv_epi32(fraction, shift);
  constexpr int bases[] = {0, 175, 330, 471, 599, 717, 827, 929},
                slopes[] = {176, 155, 142, 129, 119, 110, 102, 95};
  const auto base = _mm_i32gather_epi32(bases, segment, 4),
             slope = _mm_i32gather_epi32(slopes, segment, 4);
  const auto remainder = _mm_sub_epi32(fraction, _mm_sllv_epi32(segment, shift));
  auto sum = _mm_add_epi32(_mm_sllv_epi32(base, shift), _mm_mullo_epi32(slope, remainder));
  sum = _mm_add_epi32(sum, _mm_slli_epi32(denominator, 8));
  const auto log = _mm_srlv_epi32(sum, _mm_add_epi32(leading, _mm_set1_epi32(9)));
  const auto whole =
      _mm_slli_epi32(_mm_add_epi32(exponent, _mm_sub_epi32(leading, _mm_set1_epi32(11))), 8);
  return _mm_blendv_epi8(_mm_add_epi32(whole, log), _mm_set1_epi32(zero_lod),
                         _mm_cmpeq_epi32(value, zero()));
}
// Evaluate normalized gradient precision stages, then SIMD integer norms/logarithms.
inline void footprints(int count, const rj_arith_image_gradients *input, const uint32_t *extents,
                       int dimensions, ImageLodAxes *output) {
  int extent_exponents[3]{};
  double extent_mantissas[3]{};
  for (int axis = 0; axis < dimensions; ++axis) {
    extent_exponents[axis] = bit_width(extents[axis]) - 1;
    extent_mantissas[axis] =
        __builtin_floor(__builtin_ldexp(double(extents[axis]), 10 - extent_exponents[axis])) / 1024;
  }
  for (int start = 0; start < count; start += 4) {
    unsigned repair = 0;
    D scaled[6]{};
    I coordinate_exponents[3]{};
    auto common = _mm_set1_epi32(-2048);
    for (int axis = 0; axis < dimensions; ++axis) {
      D gradients[2];
      for (int half = 0; half < 2; ++half) {
        alignas(64) double values[4];
        RJ_ARITH_UNROLL(4)
        for (int i = 0; i < 4; ++i) {
          values[i] = half ? input[start + i].dy[axis] : input[start + i].dx[axis];
        }
        auto value = _mm256_load_pd(values);
        value = select(value,
                       _mm256_or_pd(_mm256_and_pd(value, _mm256_set1_pd(-0.0)),
                                    _mm256_set1_pd(double(0x1.fffffep127f))),
                       compare<_CMP_EQ_OQ>(magnitude(value), _mm256_set1_pd(__builtin_inf())));
        value = select(value, _mm256_setzero_pd(), compare<_CMP_UNORD_Q>(value, value));
        // Coordinate differences normally fit the FP32 range. Keep arbitrary
        // tiny FP64 gradients correct without constructing overflowing scales.
        auto tiny = _mm256_and_pd(compare<_CMP_LT_OQ>(magnitude(value), _mm256_set1_pd(0x1p-1000)),
                                  compare<_CMP_NEQ_OQ>(value, _mm256_setzero_pd()));
        repair |= mask_bits(tiny);
        gradients[half] = select(value, _mm256_setzero_pd(), tiny);
      }
      const auto largest = _mm256_max_pd(magnitude(gradients[0]), magnitude(gradients[1]));
      const auto exponent = exponents(largest);
      coordinate_exponents[axis] = _mm_add_epi32(exponent, _mm_set1_epi32(extent_exponents[axis]));
      common = _mm_max_epi32(common, coordinate_exponents[axis]);
      const auto nonzero_exponent =
          _mm_blendv_epi8(exponent, zero(), _mm_cmpeq_epi32(exponent, _mm_set1_epi32(-1022)));
      const auto scale = power_two(_mm_sub_epi32(_mm_set1_epi32(11), nonzero_exponent));
      for (int half = 0; half < 2; ++half) {
        auto mantissa = _mm256_mul_pd(magnitude(gradients[half]), scale);
        auto significand = floor(_mm256_add_pd(mantissa, _mm256_set1_pd(0.5)));
        if (dimensions != 3) {
          alignas(64) int mask[4]{};
          bool any = false;
          for (int i = 0; i < 4; ++i) {
            const auto direction = input[start + i].rounding[half * 2 + axis];
            const double v = half ? input[start + i].dy[axis] : input[start + i].dx[axis];
            const bool toward_zero = direction && ((direction < 0) != bool(__builtin_signbit(v)));
            mask[i] = toward_zero ? -1 : 0;
            any |= toward_zero;
          }
          if (any) {
            const auto selected = from_bits(_mm256_cvtepi32_epi64(load(mask)));
            const auto ceil = _mm256_sub_pd(_mm256_setzero_pd(),
                                            floor(_mm256_sub_pd(_mm256_set1_pd(0.5), mantissa)));
            significand = select(significand, ceil, selected);
          }
        }
        significand = _mm256_min_pd(significand, _mm256_set1_pd(2047));
        significand =
            _mm256_or_pd(_mm256_and_pd(gradients[half], _mm256_set1_pd(-0.0)), significand);
        scaled[half * 3 + axis] =
            floor(_mm256_add_pd(_mm256_mul_pd(significand, _mm256_set1_pd(extent_mantissas[axis])),
                                _mm256_set1_pd(0.5)));
      }
    }
    I fixed[6];
    for (int i = 0; i < 6; ++i) {
      fixed[i] = zero();
    }
    for (int axis = 0; axis < dimensions; ++axis) {
      const auto delta =
          _mm_max_epi32(_mm_sub_epi32(coordinate_exponents[axis], common), _mm_set1_epi32(-1022));
      const auto scale = power_two(delta);
      for (int half = 0; half < 2; ++half) {
        fixed[half * 3 + axis] = _mm256_cvttpd_epi32(floor(
            _mm256_add_pd(_mm256_mul_pd(scaled[half * 3 + axis], scale), _mm256_set1_pd(0.5))));
      }
    }
    I first, second, major, minor;
    if (dimensions == 3) {
      first = norm(norm(fixed[0], fixed[1]), fixed[2]);
      second = norm(norm(fixed[3], fixed[4]), fixed[5]);
      major = _mm_max_epi32(first, second);
      minor = zero();
    } else {
      first = norm(_mm_add_epi32(fixed[0], fixed[4]), _mm_sub_epi32(fixed[1], fixed[3]));
      second = norm(_mm_sub_epi32(fixed[0], fixed[4]), _mm_add_epi32(fixed[1], fixed[3]));
      major = _mm_srli_epi32(_mm_add_epi32(first, second), 1);
      minor = _mm_abs_epi32(_mm_srai_epi32(_mm_sub_epi32(first, second), 1));
    }
    const auto major_lod = logarithm(major, common, -8192),
               minor_lod = logarithm(minor, common, -16384);
    alignas(64) int32_t norms[2][4], lods[2][4];
    store(norms[0], first);
    store(norms[1], second);
    store(lods[0], major_lod);
    store(lods[1], minor_lod);
    for (int i = 0; i < 4; ++i) {
      for (int component = 0; component < 2; ++component) {
        output[start + i].norms[component] = uint32_t(norms[component][i]);
        output[start + i].lods[component] = lods[component][i];
      }
    }
    for (; repair; repair &= repair - 1) {
      const int lane = __builtin_ctz(repair);
      const auto scalar = image_lod::footprint(input[start + lane], extents, dimensions);
      for (int i = 0; i < 2; ++i) {
        output[start + lane].norms[i] = scalar.norms[i];
        output[start + lane].lods[i] = scalar.lods[i];
      }
    }
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod_simd
