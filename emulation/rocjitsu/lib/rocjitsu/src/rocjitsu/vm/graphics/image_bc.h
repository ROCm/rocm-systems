// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GRAPHICS_IMAGE_BC_H_
#define ROCJITSU_VM_GRAPHICS_IMAGE_BC_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

// BC3 alpha and BC4/5 share selectors and six-bit interpolation weights.
// Retain the fractional bits for BC4/5; BC3 quantizes this result to UNORM8.
inline int32_t decode_image_bc_scalar(std::span<const uint8_t> block, uint32_t texel,
                                      bool signed_format = false) {
  const auto endpoint = [&](uint8_t value) {
    return signed_format ? std::max(-127, value < 128 ? int32_t(value) : int32_t(value) - 256)
                         : int32_t(value);
  };
  const int32_t a = endpoint(block[0]), b = endpoint(block[1]);
  const uint32_t bit = 3 * texel, byte = 2 + bit / 8;
  const uint32_t pair = block[byte] | (byte < 7 ? uint32_t(block[byte + 1]) << 8 : 0);
  const uint32_t index = (pair >> (bit % 8)) & 7;
  if (index < 2)
    return (index ? b : a) * 64;
  if (a <= b && index >= 6)
    return (index == 7 ? (signed_format ? 127 : 255) : (signed_format ? -127 : 0)) * 64;
  constexpr int32_t weights7[]{9, 18, 27, 37, 46, 55};
  constexpr int32_t weights5[]{13, 26, 38, 51};
  const int32_t weight = a > b ? weights7[index - 2] : weights5[index - 2];
  return (64 - weight) * a + weight * b;
}

/// Decode one BC1/2/3 texel to RGBA8. RDNA3/4 use rounded six-bit
/// interpolation weights, rather than exact division by three, five or seven.
inline std::array<uint8_t, 4> decode_image_bc(uint32_t format, uint32_t texel,
                                              std::span<const uint8_t> block) {
  std::array<uint8_t, 4> out{};
  if (block.empty())
    return out;
  const uint32_t kind = (format - 109) / 2;
  const auto color = block.subspan(kind ? 8 : 0);
  const uint32_t endpoints[]{uint32_t(color[0]) | (uint32_t(color[1]) << 8),
                             uint32_t(color[2]) | (uint32_t(color[3]) << 8)};
  const uint32_t index = (color[4 + texel / 4] >> (2 * (texel % 4))) & 3;
  const bool three_colors = kind == 0 && endpoints[0] <= endpoints[1];
  for (uint32_t c = 0; c < 3; ++c) {
    const uint32_t shift = c == 0 ? 11 : c == 1 ? 5 : 0;
    const uint32_t width = c == 1 ? 6 : 5;
    const auto expand = [&](uint32_t endpoint) {
      const uint32_t value = (endpoint >> shift) & ((1u << width) - 1);
      return (value << (8 - width)) | (value >> (2 * width - 8));
    };
    const uint32_t a = expand(endpoints[0]), b = expand(endpoints[1]);
    out[c] = index < 2      ? (index ? b : a)
             : three_colors ? (index == 2 ? (a + b + 1) / 2 : 0)
             : index == 2   ? (43 * a + 21 * b + 32) / 64
                            : (21 * a + 43 * b + 32) / 64;
  }
  out[3] = three_colors && index == 3 ? 0 : 255;
  if (kind == 1) {
    out[3] = ((block[texel / 2] >> (4 * (texel % 2))) & 15) * 17;
  } else if (kind == 2) {
    out[3] = (decode_image_bc_scalar(block, texel) + 32) / 64;
  }
  return out;
}

} // namespace rocjitsu::amdgpu

#endif
