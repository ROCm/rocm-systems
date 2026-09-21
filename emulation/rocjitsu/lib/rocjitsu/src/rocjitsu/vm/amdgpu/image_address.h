// Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_ADDRESS_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_ADDRESS_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// GFX12 non-MSAA, single-level 2D surface addressing.
/// The bit patterns match AddrLib's gfx12SwizzlePattern.h for 1xAA 2D blocks.
inline std::optional<uint64_t> gfx12_image_offset(uint32_t x, uint32_t y, uint32_t width,
                                                  uint32_t bytes, uint32_t swizzle) {
  if (!width || x >= width || (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 16))
    return std::nullopt;
  if (!swizzle)
    return (uint64_t{y} * width + x) * bytes;
  if (swizzle > 4)
    return std::nullopt;
  // Positive selectors address x, negative selectors address y; zero is a byte bit.
  static constexpr std::array<std::array<int8_t, 18>, 5> patterns{{
      {{1, 2, -1, 3, -2, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9}},
      {{0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9}},
      {{0, 0, 1, -1, 2, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8}},
      {{0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8}},
      {{0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7}},
  }};
  uint32_t element_log2 = 0;
  while ((1u << element_log2) < bytes)
    ++element_log2;
  const uint32_t block_log2 = swizzle == 4 ? 18 : 4 * swizzle + 4;
  uint32_t x_bits = 0, y_bits = 0, offset = 0;
  for (uint32_t bit = 0; bit < block_log2; ++bit) {
    const int selector = patterns[element_log2][bit];
    if (selector > 0) {
      x_bits = std::max(x_bits, uint32_t(selector));
      offset |= ((x >> (selector - 1)) & 1u) << bit;
    } else if (selector < 0) {
      y_bits = std::max(y_bits, uint32_t(-selector));
      offset |= ((y >> (-selector - 1)) & 1u) << bit;
    }
  }
  const uint32_t pitch_blocks = (width + (1u << x_bits) - 1) >> x_bits;
  const uint64_t block = uint64_t{y >> y_bits} * pitch_blocks + (x >> x_bits);
  return (block << block_log2) + offset;
}

/// Tiled bases include pipe/bank XOR bits in their low, block-relative address.
inline std::optional<uint64_t> gfx12_image_address(uint64_t base, uint32_t x, uint32_t y,
                                                   uint32_t width, uint32_t bytes,
                                                   uint32_t swizzle) {
  const auto offset = gfx12_image_offset(x, y, width, bytes, swizzle);
  if (!offset)
    return std::nullopt;
  if (!swizzle)
    return base + *offset;
  const uint32_t block_log2 = swizzle == 4 ? 18 : 4 * swizzle + 4;
  const uint64_t block_mask = (uint64_t{1} << block_log2) - 1;
  return (base & ~block_mask) + (*offset ^ (base & block_mask));
}

} // namespace rocjitsu::amdgpu

#endif
