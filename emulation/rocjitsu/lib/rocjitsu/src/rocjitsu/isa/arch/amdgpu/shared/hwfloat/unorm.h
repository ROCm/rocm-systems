// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>

namespace rocjitsu::amdgpu::hwfloat {

constexpr bool supports_unorm_width(uint32_t width) {
  return width == 2 || width == 8 || width == 10 || width == 16;
}

/// Clamp binary32 to [0, 1] and quantize to UNORM with nearest-even rounding.
/// NaNs and negative values map to zero. Uses no host FP operations or state.
/// Only the listed widths are supported; wider formats retain their FP path.
constexpr uint32_t unorm_from_f32(uint32_t bits, uint32_t width) {
  assert(supports_unorm_width(width));
  const uint32_t maximum = (uint32_t{1} << width) - 1;
  const uint32_t magnitude = bits & 0x7fffffff;
  if ((bits >> 31) || magnitude > 0x7f800000)
    return 0;
  if (magnitude >= 0x3f800000)
    return maximum;
  const uint32_t exponent = magnitude >> 23;
  // Below 2^(-width-1), the result rounds to zero, including all subnormals.
  if (exponent + width < 126)
    return 0;
  // x = significand * 2^(exponent-150). The product uses at most 40 bits.
  const uint64_t product = uint64_t{(magnitude & 0x7fffff) | 0x800000} * maximum;
  const uint32_t shift = 150 - exponent; // 24..(24+width), hence at most 40.
  const uint32_t quotient = static_cast<uint32_t>(product >> shift);
  const uint64_t remainder = product & ((uint64_t{1} << shift) - 1);
  const uint64_t halfway = uint64_t{1} << (shift - 1);
  return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}

} // namespace rocjitsu::amdgpu::hwfloat
