// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 cube-map face selection, coordinates, and major-axis magnitude.
// CPU-specific arithmetic stays here; public entry points select this backend.

// Adapted from rocjitsu's shared/cube.h bit-level model.

#include "graphics/cube.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

using Mask = __m256i;

// Intersect lane predicates.
Mask both(Mask a, Mask b) { return _mm256_and_si256(a, b); }

// Complement a lane predicate.
Mask invert(Mask a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

// Return the lane mask for a > b.
Mask greater(__m256i a, __m256i b) { return _mm256_cmpgt_epi32(a, b); }

// Return the lane mask for equal operands.
Mask equal(__m256i a, __m256i b) { return _mm256_cmpeq_epi32(a, b); }

// Select values lane by lane according to the supplied predicate.
__m256i select(Mask exec_mask, __m256i yes, __m256i no) {
  return _mm256_blendv_epi8(no, yes, exec_mask);
}

// Return the per-lane predicate for a <= b.
Mask less_equal(__m256i a, __m256i b) { return invert(greater(a, b)); }

// Return absolute FP32 bits, mapping subnormal magnitudes to zero.
__m256i magnitude(__m256i raw) {
  auto mag = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MAX));
  return select(greater(_mm256_set1_epi32(0x00800000), mag), _mm256_setzero_si256(), mag);
}

// rocjitsu's directed overflow, expressed with the existing SIMD selectors.
__m256i overflow(__m256i sign, unsigned round_mode) {
  auto magnitude = _mm256_set1_epi32(0x7f800000);
  const auto finite = _mm256_set1_epi32(0x7f7fffff);
  const auto zero = _mm256_setzero_si256();
  if (round_mode == 1) {
    magnitude = select(invert(equal(sign, zero)), finite, magnitude);
  } else if (round_mode == 2) {
    magnitude = select(equal(sign, zero), finite, magnitude);
  } else if (round_mode == 3) {
    magnitude = finite;
  }
  return _mm256_or_si256(sign, magnitude);
}

} // namespace

namespace rocjitsu::arith {

// Evaluate cube-map coordinates or face selection. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Cube Op>
void cube_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                    const uint32_t *b, const uint32_t *c, bool quiet_nan, bool clamp_nan,
                    unsigned round_mode) {
  auto keep_a = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto keep_b = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  auto keep_c = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_C ? INT32_MAX : -1);
  auto flip_a = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto flip_b = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  auto flip_c = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_C ? INT32_MIN : 0);
  auto zero = _mm256_setzero_si256(), sign_bit = _mm256_set1_epi32(INT32_MIN);
  auto infinity = _mm256_set1_epi32(0x7f800000), step = _mm256_set1_epi32(0x00800000);
  unsigned omod = (mode >> 6) & 3;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    x = _mm256_xor_si256(_mm256_and_si256(x, keep_a), flip_a);
    y = _mm256_xor_si256(_mm256_and_si256(y, keep_b), flip_b);
    z = _mm256_xor_si256(_mm256_and_si256(z, keep_c), flip_c);
    auto ax = magnitude(x), ay = magnitude(y), az = magnitude(z);
    auto ordered_xy = both(less_equal(ax, infinity), less_equal(ay, infinity));
    auto z_axis = both(both(ordered_xy, less_equal(az, infinity)),
                       both(less_equal(ax, az), less_equal(ay, az)));
    auto y_axis = both(ordered_xy, less_equal(ax, ay));
    auto major = select(z_axis, z, select(y_axis, y, x));
    auto mag = magnitude(major);
    auto negative = both(greater(zero, major), both(greater(mag, zero), less_equal(mag, infinity)));
    auto sign = select(negative, sign_bit, zero);
    __m256i result;
    if constexpr (Op == Cube::Id) {
      auto face_z = select(negative, _mm256_set1_epi32(0x40a00000), _mm256_set1_epi32(0x40800000));
      auto face_y = select(negative, _mm256_set1_epi32(0x40400000), _mm256_set1_epi32(0x40000000));
      auto face_x = select(negative, _mm256_set1_epi32(0x3f800000), zero);
      result = select(z_axis, face_z, select(y_axis, face_y, face_x));
    } else if constexpr (Op == Cube::Sc) {
      result = select(z_axis, _mm256_xor_si256(x, sign),
                      select(y_axis, x, _mm256_xor_si256(z, _mm256_xor_si256(sign, sign_bit))));
    } else if constexpr (Op == Cube::Tc) {
      result = select(both(invert(z_axis), y_axis), _mm256_xor_si256(z, sign),
                      _mm256_xor_si256(y, sign_bit));
    } else {
      auto overflow_value = overflow(_mm256_and_si256(major, sign_bit), round_mode);
      result = select(less_equal(_mm256_set1_epi32(0x7f000000), mag), overflow_value,
                      _mm256_add_epi32(major, step));
      result = select(less_equal(infinity, mag), major, result);
      result = select(equal(mag, zero), round_mode == 2 ? _mm256_and_si256(major, sign_bit) : zero,
                      result);
    }
    auto nan = greater(_mm256_and_si256(result, _mm256_set1_epi32(INT32_MAX)), infinity);
    result =
        select(nan, _mm256_or_si256(result, _mm256_set1_epi32(quiet_nan ? 0x00400000 : 0)), result);
    if (omod) {
      auto exponent = _mm256_and_si256(result, infinity);
      auto output_sign = _mm256_and_si256(result, sign_bit);
      __m256i scaled;
      if (omod == 3) {
        scaled = select(equal(exponent, step), output_sign, _mm256_sub_epi32(result, step));
      } else {
        auto increment = _mm256_set1_epi32(int(omod * 0x00800000));
        scaled = select(less_equal(_mm256_sub_epi32(infinity, increment), exponent),
                        overflow(output_sign, round_mode), _mm256_add_epi32(result, increment));
      }
      result = select(equal(exponent, infinity), result, scaled);
      result = select(equal(exponent, zero), zero, result);
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto unclamped = result;
      auto positive = select(greater(result, infinity), zero, result);
      result = _mm256_min_epi32(_mm256_max_epi32(positive, zero), _mm256_set1_epi32(0x3f800000));
      if (!clamp_nan) {
        result = select(nan, unclamped, result);
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void cube_x86_64_v3<Cube::Id>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *, bool, bool, unsigned);
template void cube_x86_64_v3<Cube::Sc>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *, bool, bool, unsigned);
template void cube_x86_64_v3<Cube::Tc>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *, bool, bool, unsigned);
template void cube_x86_64_v3<Cube::Ma>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                       const uint32_t *, const uint32_t *, bool, bool, unsigned);

} // namespace rocjitsu::arith
