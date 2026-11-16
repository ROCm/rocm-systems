// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Borrowed empirical FP32 sine/cosine model and fixed-point range reduction.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_TRIG_MODEL_H_
#define RJ_ARITH_TRIG_MODEL_H_

/// @file
/// Integer SIN/COS reduction and staged approximation from RDNA3/4 captures.
/// Adapted from rocjitsu lib/util/include/util/amdgpu_trig.h.

#include "common/bits.h"
#include "common/uint128.h"
#include "math/trig_coefficients.h"

#include <cstdint>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig {

using U128 = Uint128;

// Return the number of significant bits, with zero mapped to zero.
inline unsigned bit_width(uint64_t value) {
  unsigned width = 0;
  while (value) {
    ++width;
    value >>= 1;
  }
  return width;
}

struct Coefficient {
  int32_t constant;
  int32_t linear;
  int32_t quadratic;
  int32_t cubic;
};

// Round an unsigned fixed-point integer right shift to nearest, ties to even.
inline uint64_t round_even(U128 value, unsigned shift) {
  if (!shift) {
    return static_cast<uint64_t>(value);
  }
  U128 whole = value >> shift;
  U128 rest = value & ((U128{1} << shift) - 1);
  U128 midpoint = U128{1} << (shift - 1);
  return static_cast<uint64_t>(whole +
                               (rest > midpoint || (rest == midpoint && ((whole & 1) != 0))));
}

// Convert a positive fixed-point result to FP32 with truncation.
inline uint32_t fixed_to_float(uint64_t value, unsigned fractional) {
  if (!value) {
    return 0;
  }
  const unsigned leading = static_cast<unsigned>(bit_width(value)) - 1;
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
  return rocjitsu::arith::minimum(rounded, limit) << shift;
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
  if (!product) {
    return bits & 0x80000000u;
  }
  const unsigned leading = static_cast<unsigned>(bit_width(product)) - 1;
  int result_exponent = exponent - 44 + static_cast<int>(leading);
  uint32_t result;
  if (result_exponent < -126) {
    result = static_cast<uint32_t>(round_even(product, 21));
    if (!(denorm & 2) && result < 0x800000) {
      result = 0;
    }
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

// Return empirical sine/cosine bits with the selected guest denormal and sNaN policy.
inline uint32_t evaluate(uint32_t bits, bool cosine, unsigned denorm = 3, bool quiet_snan = true) {
  uint32_t magnitude = bits & 0x7fffffff;
  if (magnitude >= 0x7f800000) {
    if (magnitude == 0x7f800000) {
      return 0xffc00000;
    }
    return bits | (quiet_snan ? 0x00400000u : 0);
  }
  if (!magnitude || (magnitude < 0x00800000 && !(denorm & 1))) {
    return cosine ? 0x3f800000 : bits & 0x80000000u;
  }
  if (!cosine && magnitude < 0x39c00000) {
    return small_sine(bits, denorm);
  }

  // Reduce finite inputs in turns before polynomial evaluation. Every FP32
  // value with exponent >= 23 is integral, so no large float-to-int cast is
  // needed. The phase retains 31 fractional bits.
  int exponent = static_cast<int>(magnitude >> 23) - 127;
  if (exponent >= 23) {
    return cosine ? 0x3f800000 : 0;
  }
  uint64_t mantissa = (magnitude & 0x7fffff) | 0x800000;
  int shift = exponent + 8;
  uint32_t phase = static_cast<uint32_t>(shift >= 0     ? (mantissa << shift) & 0x7fffffff
                                         : -shift >= 64 ? 0
                                                        : mantissa >> -shift);
  unsigned quadrant = phase >> 29;
  uint32_t quarter_phase = phase & 0x1fffffff;
  bool reflected = quarter_phase >= 0x10000000;
  uint32_t reduced = reflected ? 0x20000000 - quarter_phase : quarter_phase;
  int table = unsigned(cosine) ^ (quadrant & 1) ^ unsigned(reflected);
  int index = (reduced - unsigned(reflected)) >> 24;
  uint32_t fraction = reduced - (index << 24);
  const int column = table * 16 + index;
  const Coefficient c = {coefficients[0][column], coefficients[1][column], coefficients[2][column],
                         coefficients[3][column]};
  uint32_t result;
  if (table && reduced - unsigned(reflected) < 46592) {
    result = 0x3f800000;
  } else if (!table && !reduced) {
    result = 0;
  } else {
    uint32_t square_input = (fraction - unsigned(reflected)) >> 7;

    // Every interval has a nonpositive quadratic correction. Keep its
    // magnitude unsigned so the wide-integer fallback follows the same path.
    // The inner coefficient retains eighteen coordinate bits. Cosine uses
    // the ones-complement distance from the right endpoint. Round the
    // coefficient to seven fractional bits in its table units (Q33 overall).
    uint32_t inner_coordinate = c.cubic <= 0 ? fraction - unsigned(reflected)
                                             : (1u << 24) - fraction + unsigned(reflected) - 1;
    inner_coordinate &= ~63u;
    uint64_t inner_value =
        c.cubic <= 0 ? static_cast<uint64_t>(-int64_t{c.quadratic} * (int64_t{1} << 24) -
                                             int64_t{c.cubic} * inner_coordinate)
                     : static_cast<uint64_t>(-int64_t{c.quadratic + c.cubic} * (int64_t{1} << 24) +
                                             int64_t{c.cubic} * inner_coordinate);
    uint64_t staged_inner = round_even(inner_value, 17);

    // Sine near zero retains more product bits. The zero-origin interval
    // normalizes in groups of four bits to cover reduced phases near zero.
    if (!table && index < 2) {
      unsigned width = static_cast<unsigned>(bit_width(fraction));
      unsigned normalization = index ? 0 : 4 * (width < 24 ? (24 - width) / 4 : 0);
      unsigned offset_shift = index ? 0 : 4 * (width < 25 ? (25 - width) / 4 : 0);
      uint64_t linear_input = index ? uint64_t{linear_fraction(fraction, 2, reflected)}
                                    : uint64_t{fraction} << offset_shift;

      // Saturation follows the normalized field, even when the linear
      // input uses an additional four offset bits.
      if (!index && reflected && ((uint64_t{fraction} << normalization) & 0xfffff) == 0) {
        --linear_input;
      }
      unsigned linear_precision = index ? 30 : 32 + normalization;
      unsigned quadratic_precision = 34 + normalization;
      int64_t linear =
          static_cast<int64_t>((uint64_t{static_cast<uint32_t>(c.linear)} * linear_input) >>
                               (50 + offset_shift - linear_precision));

      // Truncate the square before multiplying by the rounded coefficient.
      unsigned square_precision = rocjitsu::arith::minimum(34u, 24 + normalization);
      uint64_t staged_square = (uint64_t{square_input} * square_input) >> (34 - square_precision);
      int64_t quadratic = -static_cast<int64_t>(round_even(
          U128{staged_inner} * staged_square, 33 + square_precision - quadratic_precision));
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
      uint64_t staged_square = (uint64_t{square_input} * square_input) >> 10;
      uint64_t quadratic_product = staged_inner * staged_square;

      // Round the quadratic product to 34 fractional bits, capped at
      // 24 significant bits, before flooring the negative correction to Q28.
      const uint64_t bias = uint64_t{1} << (quadratic_product >= (uint64_t{1} << 47) ? 23 : 22);
      int64_t quadratic =
          quadratic_product <= bias
              ? 0
              : -static_cast<int64_t>((quadratic_product - bias + (uint64_t{1} << 29) - 1) >> 29);
      int64_t sum = int64_t{c.constant} * 8 + linear + quadratic * 2;
      result = fixed_to_float(static_cast<uint64_t>(sum), 29);
    }
  }
  if (!result) {
    return 0;
  }
  bool negative = (quadrant >> 1) ^ (unsigned(cosine) & quadrant & 1) ^ (!cosine && (bits >> 31));
  return result | (uint32_t{negative} << 31);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig

#endif
