// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/big_int.h"
#include <algorithm>
#include <bit>
#include <cstdint>

namespace util::detail::trig {
using U128 = uint128_t;
struct Coefficient {
  int32_t constant, linear, quadratic, cubic;
};
// Coefficients use units of 2^-26. The first table approximates sine, the
// second cosine, in sixteen intervals covering the first octant. Staged
// integer arithmetic follows captures from gfx1100 and gfx1201; it avoids
// dependence on host rounding, denormal controls, or libm argument reduction.
// The polynomial is not an exhaustive bit-exact hardware mapping.
inline constexpr Coefficient coefficients[2][16] = {{{0, 3294200, 0, -1322},
                                                     {3292874, 3290232, -3968, -1320},
                                                     {6577818, 3278340, -7928, -1312},
                                                     {9846914, 3258548, -11864, -1304},
                                                     {13092288, 3230908, -15778, -1290},
                                                     {16306122, 3195480, -19650, -1274},
                                                     {19480674, 3152356, -23476, -1256},
                                                     {22608294, 3101636, -27246, -1234},
                                                     {25681450, 3043444, -30948, -1210},
                                                     {28692736, 2977924, -34576, -1182},
                                                     {31634898, 2905224, -38122, -1152},
                                                     {34500850, 2825532, -41576, -1118},
                                                     {37283686, 2739032, -44928, -1082},
                                                     {39976702, 2645928, -48176, -1042},
                                                     {42573412, 2546452, -51304, -1002},
                                                     {45067558, 2440844, -54310, -958}},
                                                    {{67108864, 0, -80872, 32},
                                                     {67028028, -161636, -80774, 98},
                                                     {66785716, -322884, -80482, 162},
                                                     {66382510, -483352, -79996, 226},
                                                     {65819384, -642664, -79318, 290},
                                                     {65097694, -800416, -78448, 352},
                                                     {64219178, -956248, -77390, 414},
                                                     {63185952, -1109776, -76144, 476},
                                                     {62000504, -1260628, -74716, 536},
                                                     {60665694, -1408448, -73104, 594},
                                                     {59184734, -1552872, -71320, 652},
                                                     {57561192, -1693552, -69364, 708},
                                                     {55798980, -1830152, -67242, 762},
                                                     {53902344, -1962348, -64956, 814},
                                                     {51875852, -2089812, -62514, 864},
                                                     {49724388, -2212244, -59922, 912}}};

inline uint64_t round_even(U128 value, unsigned shift) {
  if (!shift)
    return static_cast<uint64_t>(value);
  U128 whole = value >> shift;
  U128 rest = value & ((U128{1} << shift) - 1);
  U128 midpoint = U128{1} << (shift - 1);
  return static_cast<uint64_t>(whole +
                               (rest > midpoint || (rest == midpoint && ((whole & 1) != 0))));
}
// Convert a positive fixed-point result to FP32 with truncation.
inline uint32_t fixed_to_float(uint64_t value, unsigned fractional) {
  if (!value)
    return 0;
  const unsigned leading = static_cast<unsigned>(std::bit_width(value)) - 1;
  const uint32_t mantissa =
      static_cast<uint32_t>(leading >= 23 ? value >> (leading - 23) : value << (23 - leading));
  const int exponent = static_cast<int>(leading) - static_cast<int>(fractional) + 127;
  return (static_cast<uint32_t>(exponent) << 23) | (mantissa & 0x7fffff);
}
// Round the linear coordinate while saturating the low twenty bits.
// Reflected intervals approach their right endpoint from below.
inline uint32_t linear_fraction(uint32_t fraction, unsigned shift, bool reflected) {
  uint32_t value = fraction - static_cast<uint32_t>(reflected);
  uint32_t block = value >> (20 + shift);
  uint32_t rounded = (value + (1u << (shift - 1))) >> shift;
  uint32_t limit = ((block + 1) << 20) - 1;
  return std::min(rounded, limit) << shift;
}
// The original small-input region uses this captured FP32 multiplier,
// independently of guest rounding. Integer multiplication also preserves
// subnormal inputs and outputs when requested by MODE.
inline uint32_t small_sine(uint32_t bits, unsigned denorm) {
  const uint32_t magnitude = bits & 0x7fffffff;
  const uint32_t exponent_bits = magnitude >> 23;
  const uint32_t mantissa = (magnitude & 0x7fffff) | (exponent_bits ? 0x800000 : 0);
  const int exponent = exponent_bits ? static_cast<int>(exponent_bits) - 127 : -126;
  const uint64_t product = uint64_t{mantissa} * 0xc90fd5u;
  if (!product)
    return bits & 0x80000000u;
  const unsigned leading = static_cast<unsigned>(std::bit_width(product)) - 1;
  int result_exponent = exponent - 44 + static_cast<int>(leading);
  uint32_t result;
  if (result_exponent < -126) {
    result = static_cast<uint32_t>(round_even(product, 21));
    if (!(denorm & 2) && result < 0x800000)
      result = 0;
  } else {
    uint32_t rounded = static_cast<uint32_t>(round_even(product, leading - 23));
    if (rounded == 0x1000000) {
      rounded >>= 1;
      ++result_exponent;
    }
    result = (uint32_t(result_exponent + 127) << 23) | (rounded & 0x7fffff);
  }
  return result | (bits & 0x80000000u);
}
inline uint32_t evaluate(uint32_t bits, bool cosine, unsigned denorm = 3, bool quiet_snan = true) {
  uint32_t magnitude = bits & 0x7fffffff;
  if (magnitude >= 0x7f800000) {
    if (magnitude == 0x7f800000)
      return 0xffc00000;
    return bits | (quiet_snan ? 0x00400000u : 0);
  }
  if (!magnitude || (magnitude < 0x00800000 && !(denorm & 1)))
    return cosine ? 0x3f800000 : bits & 0x80000000u;
  if (!cosine && magnitude < 0x39c00000)
    return small_sine(bits, denorm);
  // Reduce finite inputs in turns before polynomial evaluation. Every FP32
  // value with exponent >= 23 is integral, so no large float-to-int cast is
  // needed. The phase retains 31 fractional bits.
  int exponent = static_cast<int>(magnitude >> 23) - 127;
  if (exponent >= 23)
    return cosine ? 0x3f800000 : 0;
  uint64_t mantissa = (magnitude & 0x7fffff) | 0x800000;
  int shift = exponent + 8;
  uint32_t phase = static_cast<uint32_t>(shift >= 0     ? (mantissa << shift) & 0x7fffffff
                                         : -shift >= 64 ? 0
                                                        : mantissa >> -shift);
  unsigned quadrant = phase >> 29;
  uint32_t quarter_phase = phase & 0x1fffffff;
  bool reflected = quarter_phase >= 0x10000000;
  uint32_t reduced = reflected ? 0x20000000 - quarter_phase : quarter_phase;
  unsigned table = unsigned(cosine) ^ (quadrant & 1) ^ unsigned(reflected);
  unsigned index = (reduced - unsigned(reflected)) >> 24;
  uint32_t fraction = reduced - (index << 24);
  const auto c = coefficients[table][index];
  uint32_t result;
  if (table && reduced < 46592)
    result = 0x3f800000;
  else if (!table && !reduced)
    result = 0;
  else {
    uint32_t square_input = (fraction - unsigned(reflected)) >> 7;
    // Every interval has a nonpositive quadratic correction. Keep its
    // magnitude unsigned so the wide-integer fallback follows the same path.
    uint64_t inner = static_cast<uint64_t>(-int64_t{c.quadratic} * (int64_t{1} << 24) -
                                           int64_t{c.cubic} * fraction);
    U128 quadratic_product = U128{inner} * square_input * square_input;
    // Sine near zero retains more product bits. The zero-origin interval
    // normalizes in groups of four bits to cover reduced phases near zero.
    if (!table && index < 2) {
      unsigned width = static_cast<unsigned>(std::bit_width(fraction));
      unsigned normalization = index ? 0 : 4 * (width < 24 ? (24 - width) / 4 : 0);
      unsigned offset_shift = index ? 0 : 4 * (width < 25 ? (25 - width) / 4 : 0);
      uint64_t linear_input = index ? uint64_t{linear_fraction(fraction, 2, reflected)}
                                    : uint64_t{fraction} << offset_shift;
      if (!index && reflected && (linear_input & 0xfffff) == 0)
        --linear_input;
      unsigned linear_precision = index ? 30 : 32 + normalization;
      unsigned quadratic_precision = 34 + normalization;
      int64_t linear =
          static_cast<int64_t>((uint64_t{static_cast<uint32_t>(c.linear)} * linear_input) >>
                               (50 + offset_shift - linear_precision));
      int64_t quadratic =
          -static_cast<int64_t>(round_even(quadratic_product, 84 - quadratic_precision));
      int64_t sum = (int64_t{c.constant} << (quadratic_precision - 26)) +
                    (linear << (quadratic_precision - linear_precision)) + quadratic;
      result = fixed_to_float(static_cast<uint64_t>(sum), quadratic_precision);
    } else {
      uint32_t linear_input = linear_fraction(fraction, 4, reflected);
      int64_t product = int64_t{c.linear} * linear_input;
      uint64_t absolute = static_cast<uint64_t>(product < 0 ? -product : product);
      unsigned precision = absolute >= (uint64_t{1} << 45) ? 28 : 29;
      int64_t linear = static_cast<int64_t>(round_even(absolute, 50 - precision)) *
                       (product < 0 ? -1 : 1) * (precision == 28 ? 2 : 1);
      const U128 bias = U128{1} << 50;
      int64_t quadratic = quadratic_product <= bias
                              ? 0
                              : -static_cast<int64_t>(static_cast<uint64_t>(
                                    (quadratic_product - bias + (U128{1} << 56) - 1) >> 56));
      int64_t sum = int64_t{c.constant} * 8 + linear + quadratic * 2;
      result = fixed_to_float(static_cast<uint64_t>(sum), 29);
    }
  }
  if (!result)
    return 0;
  bool negative = (quadrant >> 1) ^ (unsigned(cosine) & quadrant & 1) ^ (!cosine && (bits >> 31));
  return result | (uint32_t{negative} << 31);
}
} // namespace util::detail::trig

namespace util {

inline float amdgpu_trig_f32(float value, bool cosine, uint32_t denorm_mode, bool quiet_snan) {
  return std::bit_cast<float>(
      detail::trig::evaluate(std::bit_cast<uint32_t>(value), cosine, denorm_mode, quiet_snan));
}

} // namespace util
