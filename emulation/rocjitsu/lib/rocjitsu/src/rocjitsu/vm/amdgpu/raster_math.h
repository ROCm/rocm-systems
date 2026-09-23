// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu::raster {

inline float truncate_float(double value) {
  float result = static_cast<float>(value);
  if (std::abs(static_cast<double>(result)) > std::abs(value))
    result = std::bit_cast<float>(std::bit_cast<uint32_t>(result) - 1);
  return result;
}

// Vertex setup and fragment W use a linear reciprocal seed followed by a
// fixed-point Newton step. These coefficients and rounding stages match every
// normalized FP32 mantissa captured on physical RDNA3 and RDNA4. Shader
// reciprocals use a different approximation.
inline float reciprocal(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u || magnitude >= 0x7f800000u)
    return 1.0f / value;
  struct Coefficient {
    uint16_t base;
    uint8_t slope;
  };
  static constexpr Coefficient coefficients[] = {
      {32767, 252}, {32262, 244}, {31774, 237}, {31299, 230}, {30839, 223}, {30392, 217},
      {29958, 210}, {29536, 205}, {29126, 199}, {28727, 194}, {28338, 188}, {27961, 183},
      {27593, 179}, {27234, 174}, {26885, 170}, {26545, 165}, {26213, 161}, {25889, 157},
      {25574, 154}, {25265, 150}, {24965, 146}, {24671, 143}, {24384, 140}, {24104, 136},
      {23830, 133}, {23562, 130}, {23300, 128}, {23044, 125}, {22794, 122}, {22549, 119},
      {22309, 117}, {22074, 114}, {21844, 112}, {21619, 110}, {21398, 108}, {21182, 105},
      {20970, 103}, {20762, 101}, {20559, 99},  {20359, 97},  {20163, 96},  {19971, 94},
      {19783, 92},  {19598, 90},  {19417, 89},  {19238, 87},  {19064, 85},  {18892, 84},
      {18723, 82},  {18557, 81},  {18395, 79},  {18235, 78},  {18077, 77},  {17923, 75},
      {17771, 74},  {17622, 73},  {17475, 72},  {17330, 71},  {17188, 69},  {17049, 68},
      {16911, 67},  {16776, 66},  {16643, 65},  {16512, 64},
  };
  const uint32_t mantissa = magnitude & 0x7fffffu;
  const Coefficient coefficient = coefficients[mantissa >> 17];
  // Keep seven fraction bits and fold discarded bits into a sticky bit.
  const uint32_t fraction = ((mantissa >> 9) & 0xfeu) | ((mantissa & 0x3ffu) != 0);
  const uint32_t seed =
      (uint32_t{coefficient.base} * 128 - coefficient.slope * fraction + 128) >> 8;
  const uint32_t product = (uint64_t{mantissa | 0x800000u} * seed) >> 11;
  const uint64_t refined = uint64_t{seed} * ((1u << 27) - product);
  const uint32_t normalized = 0x3e800000u + uint32_t((refined + 0x8000u) >> 16);
  const float result = std::ldexp(std::bit_cast<float>(normalized), 127 - int(magnitude >> 23));
  return std::copysign(result, value);
}

// Setup truncates the weighted numerator before scaling by reciprocal area.
// Weighting already-divided barycentric gradients changes the low bits.
inline float plane_gradient(double edge1, double edge2, double delta1, double delta2,
                            float inverse_area) {
  const float numerator = truncate_float(edge1 * delta1 + edge2 * delta2);
  return truncate_float(double(numerator) * inverse_area);
}

// Physical RDNA3/4 perspective interpolation rounds the product using the
// input exponents, then normalizes it with truncation. A carry into another
// exponent therefore discards a bit after rounding. Midpoints round away
// from zero, unlike an IEEE float multiplication.
inline float multiply_perspective(float a, float b) {
  if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b))
    return a * b;
  const double product = double(a) * b;
  const double unit = std::ldexp(1.0, std::ilogb(a) + std::ilogb(b) - 23);
  const double rounded = std::floor(std::abs(product) / unit + 0.5) * unit;
  return truncate_float(std::copysign(rounded, product));
}

// Rasterizer quad offsets use a shared exponent and discard shifted-out bits
// before addition. This differs from rounding an IEEE addition toward zero.
inline float add_quad_offsets(float center, float dx, float dy) {
  const float largest = std::max({std::abs(center), std::abs(dx), std::abs(dy)});
  if (largest == 0)
    return 0;
  const double unit = std::ldexp(1.0, std::ilogb(largest) - 23);
  return truncate_float(
      (std::trunc(center / unit) + std::trunc(dx / unit) + std::trunc(dy / unit)) * unit);
}

struct Plane {
  float dx, dy;
  float base = 0;

  float at_quad(double x, double y, uint32_t lane) const {
    const float first = truncate_float(x * dx + y * dy + base);
    return add_quad_offsets(first, lane & 1 ? dx : 0, lane & 2 ? dy : 0);
  }
};

inline double round_subpixel(double value) {
  const double scaled = value * 256;
  const double lo = std::floor(scaled);
  const double fraction = scaled - lo;
  return (lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0))) / 256;
}

// Viewport scaling precedes perspective division. The multiplier truncates
// using its input exponents and retains a carry bit before normalization.
// Narrowing this intermediate to float loses that bit and changes coverage.
inline double multiply_viewport_scale(float position, float scale) {
  const double product = double(position) * scale;
  if (position == 0 || scale == 0 || !std::isfinite(product))
    return product;
  const double unit = std::ldexp(1.0, std::ilogb(position) + std::ilogb(scale) - 23);
  return std::trunc(product / unit) * unit;
}

enum class BlendFactorMode { Direct, One, Inverse };

inline float blend_input(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return (bits & 0x7f800000u) == 0 ? std::bit_cast<float>(bits & 0x80000000u) : value;
}

inline float blend_nan() { return std::bit_cast<float>(0xffc00000u); }

inline float blend_minmax(float a, float b, bool maximum) {
  if (std::isnan(a))
    a = b;
  else if (std::isnan(b))
    b = a;
  if (std::isnan(a))
    return blend_nan();
  const float result = maximum ? std::max(a, b) : std::min(a, b);
  return result == 0 ? 0 : result;
}

// Saturation retains the selected factor's arithmetic path. Exact ties in
// [0,1] select the inverse; other ties select the source. Opposite negative
// source/positive destination alphas stop retaining ONE after 26 bits.
inline bool blend_saturate_uses_source(float source, float destination) {
  if (std::isnan(source))
    return false;
  if (std::isnan(destination))
    return true;
  if (!std::isfinite(source) || !std::isfinite(destination))
    return source < 1.0 - double(destination);
  if (source == -destination && source <= -0x1p26f)
    return false;
  // Preserve the comparison when adding a tiny alpha would round back to one.
  if (source == 1)
    return destination < 0;
  if (destination == 1)
    return std::signbit(source);
  const double sum = double(source) + destination;
  return sum < 1 || (sum == 1 && (source < 0 || source > 1));
}

// Blending aligns product terms using their input exponents, retaining
// multiplication carries: 35 bits for FP32 and 23 for UNORM8. The built-in ONE
// factor has exponent -1; a shader alpha of 1.0 has exponent 0. Discarded bits
// remain sticky for FP32 rounding, even when the products have opposite signs.
// FP32 input subnormals must already be flushed to signed zero.
// Inverse factors take the original factor value, before subtracting from one.
inline double blend_products(float a, float af, BlendFactorMode a_mode, float b, float bf,
                             BlendFactorMode b_mode, bool negate_a, bool negate_b,
                             bool unorm = false) {
  const auto ignored = [](float factor, BlendFactorMode mode) {
    return (mode == BlendFactorMode::Direct && std::bit_cast<uint32_t>(factor) == 0) ||
           (mode == BlendFactorMode::Inverse && factor == 1);
  };
  // Positive-zero coefficients suppress even NaN and infinity colors.
  if (ignored(af, a_mode))
    a = 0;
  if (ignored(bf, b_mode))
    b = 0;
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(af) || !std::isfinite(bf)) {
    const auto product = [&](float color, float factor, BlendFactorMode mode, bool negate) {
      if (color == 0 && (mode == BlendFactorMode::One || (mode == BlendFactorMode::Inverse &&
                                                          std::bit_cast<uint32_t>(factor) == 0)))
        return 0.0;
      const double coefficient = mode == BlendFactorMode::Inverse ? 1.0 - double(factor) : factor;
      const double value = ignored(factor, mode) ? 0 : double(color) * coefficient;
      return negate ? -value : value;
    };
    const double result = product(a, af, a_mode, negate_a) + product(b, bf, b_mode, negate_b);
    return std::isnan(result) ? blend_nan() : float(result);
  }
  struct Term {
    double value;
    int exponent;
  };
  std::array<Term, 4> terms;
  unsigned count = 0;
  const auto append = [&](double value, int exponent) {
    terms[count++] = {value, value == 0 ? -1000 : exponent};
  };
  const auto product = [&](float color, float factor, BlendFactorMode mode, bool negate) {
    const float signed_color = negate ? -color : color;
    const int ec = color == 0 ? -1000 : std::ilogb(color);
    const int ef = factor == 0 ? -1000 : std::ilogb(factor);
    if (mode != BlendFactorMode::Inverse) {
      // A positive-zero coefficient clears the product sign before subtraction.
      // The built-in ONE instead clears a zero color sign after subtraction.
      const double raw = std::bit_cast<uint32_t>(factor) == 0 ? 0 : double(color) * factor;
      const double value = mode == BlendFactorMode::One && color == 0 ? 0 : negate ? -raw : raw;
      append(value, ec + (mode == BlendFactorMode::One ? -1 : ef));
    } else if (factor == 0) {
      // Unlike positive zero, a negative-zero base retains the color zero sign.
      append(color == 0 && !std::signbit(factor) ? 0 : signed_color, ec - 1);
    } else if (factor >= 0.5f && factor < 3.0f) {
      // A compact complement keeps at least the ONE factor's exponent.
      const double inverse = 1.0 - factor;
      const int ei = inverse == 0 ? -1000 : std::ilogb(inverse);
      const double raw = inverse == 0 ? 0 : double(color) * inverse;
      append(negate ? -raw : raw, ec + std::max(-1, ei));
    } else {
      // Outside that range, align the ONE and original-factor products
      // independently. Multiplying by a wide complement loses this rounding.
      append(signed_color, ec - 1);
      append(-double(signed_color) * factor, ec + ef);
    }
  };
  product(a, af, a_mode, negate_a);
  product(b, bf, b_mode, negate_b);
  int e = -1000;
  for (unsigned i = 0; i < count; ++i)
    e = std::max(e, terms[i].exponent);
  const double unit = e == -1000 ? 1 : std::ldexp(1.0, e - (unorm ? 22 : 34));
  double sum = 0;
  bool sticky = false;
  for (unsigned i = 0; i < count; ++i) {
    const double q = std::trunc(terms[i].value / unit) * unit;
    sum = i == 0 ? q : sum + q;
    sticky |= q != terms[i].value;
  }
  // UNORM quantizes the aligned sum directly to the attachment's byte value.
  // Its discarded product bits do not affect that final rounding.
  if (unorm)
    return sum;
  // Round the significand before flushing underflow; host conversion would
  // first reduce subnormal precision and can incorrectly round up to normal.
  const double scale = std::abs(sum) < 0x1p-126 ? 0x1p126 : 1;
  sum *= scale;
  float result = static_cast<float>(sum);
  if (sticky && std::abs(double(result)) < std::abs(sum)) {
    const float away = std::bit_cast<float>(std::bit_cast<uint32_t>(result) + 1);
    if (std::abs(sum) - std::abs(double(result)) == std::abs(double(away)) - std::abs(sum))
      result = away;
  }
  const double unscaled = double(result) / scale;
  return std::abs(unscaled) < 0x1p-126 ? std::copysign(0.0f, unscaled) : float(unscaled);
}

// Projection and translation each truncate to single precision before the
// fixed-point conversion. Dividing position by W before scaling changes both
// the order of operations and the precision of the scale product.
inline double viewport_coordinate(float position, float w, float scale, float offset) {
  const double scaled = multiply_viewport_scale(position, scale);
  const float projected = truncate_float(scaled * reciprocal(w));
  return round_subpixel(truncate_float(double(projected) + offset));
}

} // namespace rocjitsu::amdgpu::raster
