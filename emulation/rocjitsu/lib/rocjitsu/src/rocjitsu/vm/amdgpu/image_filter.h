// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_FILTER_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_FILTER_H_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// Texture footprint logarithm in signed units of 1/256. The GFX11/12
/// mantissa is rounded to ten fractional bits before an eight-segment log.
inline int32_t image_log2_fixed8(double value) {
  if (value <= 0)
    return -8192;
  int exponent;
  const double mantissa = std::frexp(value, &exponent);
  uint32_t fraction = static_cast<uint32_t>((2 * mantissa - 1) * 1024 + 0.5);
  if (fraction == 1024) {
    fraction = 0;
    ++exponent;
  }
  constexpr std::array<uint32_t, 8> bases{0, 175, 330, 471, 599, 717, 827, 929};
  constexpr std::array<uint32_t, 8> slopes{176, 155, 142, 128, 119, 110, 102, 95};
  const uint32_t segment = fraction >> 7;
  return (exponent - 1) * 256 +
         static_cast<int32_t>((bases[segment] * 128 + slopes[segment] * (fraction & 127) + 256) /
                              512);
}

/// Select the filter count using two-stage axis rounding and sampler controls.
/// Bias enlarges only the footprint not already covered by the anisotropy limit.
inline uint32_t image_anisotropic_filter_count(double major, double minor, uint32_t max_anisotropy,
                                               uint32_t threshold, uint32_t bias,
                                               uint32_t perf_mod) {
  if (major <= 0 || max_anisotropy == 1)
    return 1;
  // First round each axis to ten fractional significand bits, then align the
  // minor axis to the rounded major axis's exponent and round it again.
  const auto quantize = [](double value) {
    if (value <= 0)
      return 0.0;
    int exponent;
    std::frexp(value, &exponent);
    const double scale = std::ldexp(1.0, 11 - exponent);
    return std::floor(value * scale + 0.5) / scale;
  };
  major = quantize(major);
  minor = quantize(minor);
  int exponent;
  std::frexp(major, &exponent);
  const double scale = std::ldexp(1.0, 11 - exponent);
  minor = std::floor(minor * scale + 0.5) / scale;
  const int32_t major_lod = image_log2_fixed8(major);
  const int32_t minor_lod = image_log2_fixed8(minor);
  const int32_t footprint_lod =
      std::max(minor_lod, major_lod - int32_t(std::countr_zero(max_anisotropy) * 256));
  constexpr std::array<uint32_t, 8> scales{0, 2, 5, 7, 9, 11, 14, 16};
  const uint32_t perf_scale = scales[perf_mod];
  const int32_t bias_lod = static_cast<int32_t>((bias * perf_scale >> 4) * 8);
  // The anisotropy limit can already enlarge the footprint. Apply only the
  // additional enlargement due to bias, after clamping the footprint to one
  // texel. Applying all bias after that clamp miscounts sub-texel footprints.
  const int32_t biased_lod =
      major_lod - std::max(0, footprint_lod) - std::max(0, minor_lod + bias_lod - footprint_lod);
  if (biased_lod <= 0)
    return 1;
  const double fraction_threshold = (threshold * perf_scale >> 4) * 0.25;
  const double ratio = std::exp2(biased_lod / 256.0);
  // Counts above one are even. The threshold extends each lower even count's interval.
  const double count = 2 * std::ceil((ratio - fraction_threshold) / 2);
  return static_cast<uint32_t>(std::clamp(count, 1.0, double(max_anisotropy)));
}

/// Quantized, symmetric footprint weights. Residual units go to the center
/// filters, preserving a total weight of one for non-power-of-two counts.
inline double image_anisotropic_filter_weight(uint32_t count, uint32_t index) {
  const uint32_t denominator = 128 * std::bit_ceil(count);
  const uint32_t residual = denominator % count;
  const uint32_t first = (count - residual) / 2;
  const uint32_t weight = denominator / count + (index >= first && index < first + residual);
  return static_cast<double>(weight) / denominator;
}

} // namespace rocjitsu::amdgpu

#endif
