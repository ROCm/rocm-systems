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

// Clipping, depth and polygon offset flush FP32 denormals while keeping the sign.
inline float flush_denormal(float value) {
  return std::abs(value) < 0x1p-126f ? std::copysign(0.0f, value) : value;
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
  // The integer seed is normal. Scaling to another normal exponent is exact
  // in every host mode; retain ldexp for subnormal and out-of-range results.
  const int exponent = int(normalized >> 23) + 127 - int(magnitude >> 23);
  const float result =
      exponent > 0 && exponent < 255
          ? std::bit_cast<float>((normalized & 0x7fffffu) | (uint32_t(exponent) << 23))
          : std::ldexp(std::bit_cast<float>(normalized), 127 - int(magnitude >> 23));
  return std::copysign(result, value);
}

// Setup discards the smaller operand when the exponents differ by at least
// 26, then truncates the subtraction to FP32. Keeping a tiny operand would
// incorrectly borrow a low bit from the larger value.
inline float vertex_difference(float a, float b) {
  if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b))
    return a - b;
  const int difference = std::ilogb(a) - std::ilogb(b);
  if (difference >= 26)
    return a;
  if (difference <= -26)
    return -b;
  return truncate_float(double(a) - b);
}

// Attribute differences retain the input exponent through cancellation. Align
// both operands to 25 significant bits, then truncate their difference at 24
// bits of the shared scale before normalizing the resulting FP32 coefficient.
inline float attribute_difference(float a, float b) {
  // Parameter setup preserves the reference attribute's NaN before considering
  // the other vertex. It neither negates that NaN nor quiets signaling payloads.
  if ((std::bit_cast<uint32_t>(b) & 0x7fffffffu) > 0x7f800000u)
    return b;
  if ((std::bit_cast<uint32_t>(a) & 0x7fffffffu) > 0x7f800000u)
    return a;
  const auto flush = [](float value) {
    return std::abs(value) < 0x1p-126f ? std::copysign(0.0f, value) : value;
  };
  a = flush(a);
  b = flush(b);
  if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b))
    return a - b;
  const int exponent = std::max(std::ilogb(a), std::ilogb(b));
  const double unit = std::ldexp(1.0, exponent - 24);
  const double difference = std::trunc(double(a) / unit) - std::trunc(double(b) / unit);
  return truncate_float(std::trunc(difference / 2) * (2 * unit));
}

// Clipping uses independent endpoint weights. Computing one as 1 - the
// other loses the reciprocal approximation in that endpoint's contribution.
inline std::array<float, 2> clip_weights(float a_distance, float b_distance) {
  const float inverse = reciprocal(vertex_difference(a_distance, b_distance));
  return {truncate_float(-double(b_distance) * inverse),
          truncate_float(double(a_distance) * inverse)};
}

// Constant components bypass interpolation. Each product truncates to FP32;
// their sum uses the same exponent alignment as setup differences.
inline float clip_component(float a, float b, std::array<float, 2> weights) {
  if (a == b)
    return a;
  return vertex_difference(truncate_float(double(a) * weights[0]),
                           -truncate_float(double(b) * weights[1]));
}

// With PA_CL_CLIP_CNTL.DX_LINEAR_ATTR_CLIP_ENA, clipping interpolates
// noperspective barycentrics in screen space. Each endpoint weight becomes
// trunc(trunc(weight * endpoint W) * reciprocal(new W)). Zero and subnormal
// operands give a zero product, even times an infinite reciprocal.
inline std::array<float, 2> linear_clip_weights(std::array<float, 2> weights, float previous_w,
                                                float current_w, float new_w) {
  const auto multiply = [](float a, float b) {
    a = flush_denormal(a);
    b = flush_denormal(b);
    return a == 0 || b == 0 ? 0.0f : truncate_float(double(a) * b);
  };
  const float inverse = reciprocal(flush_denormal(new_w));
  return {multiply(multiply(weights[0], previous_w), inverse),
          multiply(multiply(weights[1], current_w), inverse)};
}

// Setup truncates the weighted numerator before scaling by reciprocal area.
// Weighting already-divided barycentric gradients changes the low bits.
inline float plane_numerator(double edge1, double edge2, double delta1, double delta2) {
  return truncate_float(edge1 * delta1 + edge2 * delta2);
}

inline float plane_gradient(double edge1, double edge2, double delta1, double delta2,
                            float inverse_area) {
  return truncate_float(double(plane_numerator(edge1, edge2, delta1, delta2)) * inverse_area);
}

// Primitive setup uses a wider reciprocal than vertex W. Physical RDNA3/4
// area controls identify 256 linear seed segments, nine fraction bits plus a
// sticky bit, and a truncated fixed-point Newton step. Truncate the input to
// 32 significant bits before forming the seed. Retain 32 significant bits for
// depth; ordinary interpolation truncates this result to FP32.
inline double area_reciprocal(double value) {
  if (value == 0 || !std::isfinite(value))
    return 1.0 / value;
  int exponent;
  const double normalized = std::frexp(std::abs(value), &exponent) * 2;
  const uint32_t significand = (std::bit_cast<uint64_t>(normalized) >> 21) & 0x7fffffffu;
  static constexpr uint16_t slopes[] = {
      1019, 1012, 1004, 995, 987, 980, 972, 965, 958, 952, 943, 936, 929, 922, 917, 909, 902, 896,
      889,  883,  876,  870, 864, 858, 851, 845, 839, 833, 829, 822, 816, 810, 806, 799, 795, 788,
      783,  779,  772,  767, 762, 758, 752, 748, 743, 737, 732, 727, 722, 718, 713, 708, 705, 699,
      696,  690,  686,  681, 678, 673, 668, 664, 660, 656, 653, 649, 645, 641, 636, 633, 629, 624,
      620,  618,  614,  609, 607, 602, 598, 596, 591, 589, 585, 582, 577, 575, 572, 567, 565, 562,
      557,  554,  551,  548, 546, 543, 540, 537, 534, 531, 528, 525, 522, 519, 515, 512, 509, 506,
      505,  502,  498,  496, 494, 491, 487, 486, 482, 481, 478, 474, 473, 469, 468, 464, 463, 461,
      457,  456,  452,  451, 449, 445, 444, 442, 440, 437, 434, 433, 431, 429, 425, 424, 422, 420,
      418,  416,  414,  412, 410, 408, 406, 404, 402, 400, 398, 396, 394, 392, 389, 387, 386, 385,
      383,  381,  378,  377, 374, 374, 372, 369, 367, 367, 365, 363, 362, 359, 357, 357, 355, 352,
      352,  349,  349,  346, 344, 344, 341, 341, 338, 338, 336, 335, 332, 332, 330, 329, 327, 326,
      324,  322,  322,  319, 319, 316, 316, 315, 313, 312, 311, 308, 308, 307, 304, 304, 303, 300,
      300,  299,  298,  296, 294, 294, 293, 290, 289, 289, 288, 287, 284, 284, 282, 282, 281, 280,
      277,  277,  275,  275, 274, 273, 272, 271, 270, 267, 266, 266, 265, 263, 262, 262, 261, 259,
      258,  257,  256,  256,
  };
  const uint32_t index = significand >> 23;
  const uint32_t base = (1u << 27) / (256 + index);
  const uint32_t fraction = ((significand >> 13) & 0x3feu) | ((significand & 0x3fffu) != 0);
  const uint32_t seed = (base * 512 - slopes[index] * fraction) >> 10;
  // The normalized denominator/seed product retains 32 fractional bits.
  const uint32_t mantissa = significand | 0x80000000u;
  const uint64_t product = (uint64_t{mantissa} * seed) >> 17;
  const uint64_t refined = (uint64_t{seed} * ((1ull << 33) - product)) >> 18;
  return std::copysign(std::ldexp(double(refined), -31 - exponent), value);
}

// Depth gradients truncate the FP32 numerator times the 32-bit reciprocal
// toward zero at 32 significant bits. Keep the full integer product to avoid
// a host-double rounding carry at a truncation boundary. Magnitudes below
// 2^-122, denormal at 1/16-pixel scale, flush to zero.
inline double depth_gradient(float numerator, double inverse_area) {
  if (numerator == 0 || inverse_area == 0 || !std::isfinite(numerator) ||
      !std::isfinite(inverse_area))
    return double(numerator) * inverse_area;
  int numerator_exponent, inverse_exponent;
  const double n = std::frexp(std::abs(double(numerator)), &numerator_exponent);
  const double i = std::frexp(std::abs(inverse_area), &inverse_exponent);
  const uint64_t product = uint64_t(n * 0x1p24) * uint64_t(i * 0x1p32);
  const unsigned shift = std::bit_width(product) - 32;
  const double result =
      std::ldexp(double(product >> shift), numerator_exponent + inverse_exponent + int(shift) - 56);
  return std::copysign(result < 0x1p-122 ? 0.0 : result, double(numerator) * inverse_area);
}

// Physical RDNA3/4 perspective interpolation rounds the product using the
// input exponents, then normalizes it with truncation. A carry into another
// exponent therefore discards a bit after rounding. Midpoints round away
// from zero, unlike an IEEE float multiplication.
inline float multiply_perspective(float a, float b) {
  if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b))
    return a * b;
  const double product = double(a) * b;
  const uint32_t a_exponent = (std::bit_cast<uint32_t>(a) >> 23) & 0xff;
  const uint32_t b_exponent = (std::bit_cast<uint32_t>(b) >> 23) & 0xff;
  // Two normal FP32 operands give an exact normal FP64 unit, even when
  // their product is outside the FP32 range. Keep libm for subnormal inputs.
  const double unit = a_exponent != 0 && b_exponent != 0
                          ? std::bit_cast<double>(uint64_t(a_exponent + b_exponent + 746) << 52)
                          : std::ldexp(1.0, std::ilogb(a) + std::ilogb(b) - 23);
  const double rounded = std::floor(std::abs(product) / unit + 0.5) * unit;
  return truncate_float(std::copysign(rounded, product));
}

// Rasterizer quad offsets use a shared exponent and discard shifted-out bits
// before addition. This differs from rounding an IEEE addition toward zero.
inline float add_quad_offsets(float center, float dx, float dy) {
  const float largest = std::max({std::abs(center), std::abs(dx), std::abs(dy)});
  if (largest == 0)
    return 0;
  const uint32_t magnitude = std::bit_cast<uint32_t>(largest);
  double unit;
  if (magnitude < 0x7f800000u) {
    // A finite, nonzero FP32 magnitude has exponent -149..127. Its
    // quantization unit is therefore an exact normal FP64 power of two;
    // constructing its bits avoids libm without changing the arithmetic below.
    const int exponent =
        magnitude < 0x00800000u ? int(std::bit_width(magnitude)) - 150 : int(magnitude >> 23) - 127;
    unit = std::bit_cast<double>(uint64_t(exponent + 1000) << 52);
  } else {
    const int exponent = std::ilogb(largest);
    // Keep ilogb's domain reporting for a NaN center, but do not subtract
    // from FP_ILOGBNAN: it can be INT_MIN. Propagate the center's sign and
    // payload with its quiet bit set; no quantization unit exists for NaN.
    if (magnitude > 0x7f800000u)
      return std::bit_cast<float>(std::bit_cast<uint32_t>(center) | 0x00400000u);
    unit = std::ldexp(1.0, exponent - 23);
  }
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

#if defined(__clang__) && defined(__x86_64__)
  // Inline the whole existing operation into the feature-specific caller, so
  // truncation can use SSE4.1 without a separate call for each scalar value.
  [[gnu::target("sse4.1"), gnu::noinline]] float at_quad_sse41(double x, double y,
                                                               uint32_t lane) const {
    [[clang::always_inline]] return at_quad(x, y, lane);
  }
#endif
};

// With integer pixel coordinates, these bounds keep every nonzero intermediate
// normal in at_quad_sse41. Only SSE precision flags can be added. Inspect bits
// so admission itself is inert even for signaling NaNs and subnormal inputs.
inline bool bounded_plane_pair_for_integer_pixels(const Plane &first, const Plane &second,
                                                  double origin_x, double origin_y) {
  const auto grid_origin = [](double value) {
    const uint64_t magnitude = std::bit_cast<uint64_t>(value) & 0x7fffffffffffffffull;
    if (magnitude == 0)
      return true;
    if (magnitude > 0x41d0000000000000ull) // 2^30
      return false;
    const uint32_t exponent = magnitude >> 52;
    if (exponent < 1015) // Nonzero grid points are at least 2^-8.
      return false;
    return (magnitude & ((uint64_t{1} << (1067 - exponent)) - 1)) == 0;
  };
  const auto coefficient = [](float value) {
    const uint32_t magnitude = std::bit_cast<uint32_t>(value) & 0x7fffffffu;
    return magnitude == 0 || (magnitude >= 0x30800000u && magnitude <= 0x4e800000u);
  };
  return grid_origin(origin_x) && grid_origin(origin_y) && coefficient(first.dx) &&
         coefficient(first.dy) && coefficient(first.base) && coefficient(second.dx) &&
         coefficient(second.dy) && coefficient(second.base);
}

inline bool supports_sse41_planes() {
#if defined(__clang__) && defined(__x86_64__)
  return __builtin_cpu_supports("sse4.1");
#else
  // GCC's existing truncation lowering has different precision-flag behavior.
  return false;
#endif
}

inline bool can_omit_bounded_planes(const Plane &first, const Plane &second, double origin_x,
                                    double origin_y) {
#if defined(__clang__) && defined(__x86_64__)
  if (!supports_sse41_planes() ||
      !bounded_plane_pair_for_integer_pixels(first, second, origin_x, origin_y))
    return false;
  uint16_t control;
  uint32_t mxcsr;
  // Nonwaiting reads preserve pending exceptions and raw x87 state. The
  // bounded SSE helper can only add INEXACT, which must already be sticky.
  asm volatile("fnstcw %0\n\tstmxcsr %1" : "=m"(control), "=m"(mxcsr) : : "memory");
  return (control & 0x3f) == 0x3f && (mxcsr & 0x1fa0) == 0x1fa0;
#else
  (void)first;
  (void)second;
  (void)origin_x;
  (void)origin_y;
  return false;
#endif
}

// Depth is evaluated around the center of each 8x8 tile. The center value
// and local slopes share 30 significant bits, with enough exponent range
// for either slope across a whole tile. Conversion to this signed fixed-point
// representation rounds downward, including negative slopes and tile depths.
// Before alignment, negative slopes round to 27 significant bits with ties
// away from zero, but unrounded magnitudes below a quarter of the shared unit
// shift out entirely. Retain the wider setup gradients for each new tile center
// and for choosing its shared exponent and small-slope cutoff.
struct DepthPlane {
  double dx, dy, base, origin_x, origin_y;

  float at(int x, int y) const {
    const double center_x = std::floor(x / 8.0) * 8 + 4;
    const double center_y = std::floor(y / 8.0) * 8 + 4;
    const double center = (center_x - origin_x) * dx + (center_y - origin_y) * dy + base;
    const double largest = std::max({std::abs(center), 8 * std::abs(dx), 8 * std::abs(dy)});
    if (largest == 0 || !std::isfinite(largest))
      return static_cast<float>(center);
    const auto quantization_unit = [](double value, unsigned fraction_bits) {
      const uint64_t exponent = (std::bit_cast<uint64_t>(value) >> 52) & 0x7ff;
      // A normal power of two is exact in every host mode. Keep libm for
      // subnormal units, including its range and floating-point reporting.
      if (exponent > fraction_bits && exponent < 0x7ff)
        return std::bit_cast<double>((exponent - fraction_bits) << 52);
      return std::ldexp(1.0, std::ilogb(value) - int(fraction_bits));
    };
    const double unit = quantization_unit(largest, 29);
    const double tile_base = std::floor(center / unit) * unit;
    const auto align_slope = [unit, quantization_unit](double slope) {
      if (slope < 0) {
        if (-slope < unit / 4)
          return 0.0;
        const double slope_unit = quantization_unit(slope, 26);
        slope = -std::floor(-slope / slope_unit + 0.5) * slope_unit;
      }
      return std::floor(slope / unit) * unit;
    };
    const double tile_dx = align_slope(dx);
    const double tile_dy = align_slope(dy);
    return flush_denormal(truncate_float(tile_base + (x + 0.5 - center_x) * tile_dx +
                                         (y + 0.5 - center_y) * tile_dy));
  }
};

// The polygon offset constant is scaled by 2^NEG_NUM_DB_BITS. Float depth also
// scales it by the raw FP32 exponent field of the largest vertex depth
// magnitude, so zero and denormal depths use -127 rather than -126.
inline double depth_bias_constant(float offset, uint32_t format, float largest_depth) {
  int exponent = static_cast<int8_t>(format);
  if (format & 0x100)
    exponent += int((std::bit_cast<uint32_t>(std::abs(largest_depth)) >> 23) & 0xff) - 127;
  return std::ldexp(double(offset), exponent);
}

// Physical RDNA3/4 polygon offset uses FP32 terms, and denormal terms flush to
// zero. The larger setup gradient magnitude truncates to FP32 before its
// product with the slope scale, which truncates again. A shared-exponent adder
// joins the constant term: both operands align to the larger exponent with 23
// fraction bits, shifted-out bits round toward negative infinity, and an
// operand at least 24 binades smaller contributes nothing. The sum truncates
// to FP32 before the clamp, whose zero value disables it with either sign.
inline float depth_bias(double dx, double dy, float scale, double constant, float clamp) {
  const float gradient = truncate_float(std::max(std::abs(dx), std::abs(dy)));
  const float slope = flush_denormal(truncate_float(double(gradient) * scale));
  const float offset =
      std::abs(constant) < 0x1p-126 ? std::copysign(0.0f, float(constant)) : float(constant);
  float bias = slope + offset;
  if (slope != 0 && offset != 0 && std::isfinite(bias)) {
    const int slope_exponent = std::ilogb(slope), offset_exponent = std::ilogb(offset);
    const int exponent = std::max(slope_exponent, offset_exponent);
    const auto align = [exponent](float value, int value_exponent) {
      return exponent - value_exponent >= 24 ? 0.0
                                             : std::floor(std::ldexp(double(value), 23 - exponent));
    };
    bias = flush_denormal(truncate_float(
        std::ldexp(align(slope, slope_exponent) + align(offset, offset_exponent), exponent - 23)));
  }
  if (clamp > 0)
    return std::min(bias, clamp);
  if (clamp < 0)
    return std::max(bias, clamp);
  return bias;
}

// The viewport depth clamp orders a negative zero below a positive minimum.
inline float clamp_viewport_depth(float depth, float minimum, float maximum) {
  const float clamped = std::clamp(depth, minimum, maximum);
  return std::signbit(clamped) && !std::signbit(minimum) ? minimum : clamped;
}

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

// Unaligned selects ordinary arithmetic in the caller, bypassing blend_products.
enum class BlendPrecision { Unaligned, Float32, Float16, Unorm };

// Blending aligns product terms using their input exponents, retaining
// multiplication carries: 35 bits for FP32 and 23 for FP16/UNORM. The built-in ONE
// factor has exponent -1; a shader alpha of 1.0 has exponent 0. Discarded bits
// remain sticky for FP32 and UNORM sum rounding, even when the products have
// opposite signs. FP16 quantizes the aligned sum without this rounding step.
// FP32 input subnormals must already be flushed to signed zero.
// Inverse factors take the original factor value, before subtracting from one.
// The caller must resolve Unaligned before entering this function.
inline double blend_products(float a, float af, BlendFactorMode a_mode, float b, float bf,
                             BlendFactorMode b_mode, bool negate_a, bool negate_b,
                             BlendPrecision precision) {
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
  const double unit =
      e == -1000 ? 1 : std::ldexp(1.0, e - (precision == BlendPrecision::Float32 ? 34 : 22));
  double sum = 0;
  bool sticky = false;
  for (unsigned i = 0; i < count; ++i) {
    const double q = std::trunc(terms[i].value / unit) * unit;
    sum = i == 0 ? q : sum + q;
    sticky |= q != terms[i].value;
  }
  // FP16 quantizes the aligned sum directly to the attachment. UNORM first
  // rounds to FP32, including sticky bits discarded during product alignment.
  if (precision == BlendPrecision::Float16)
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

// Projection and translation each truncate to single precision. Dividing
// position by W before scaling changes both the order of operations and the
// precision of the scale product. Depth uses this result without X/Y snapping.
inline float viewport_transform(float position, float w, float scale, float offset) {
  const double scaled = multiply_viewport_scale(position, scale);
  const float projected = truncate_float(scaled * reciprocal(w));
  return truncate_float(double(projected) + offset);
}

inline double viewport_coordinate(float position, float w, float scale, float offset) {
  return round_subpixel(viewport_transform(position, w, scale, offset));
}

} // namespace rocjitsu::amdgpu::raster
