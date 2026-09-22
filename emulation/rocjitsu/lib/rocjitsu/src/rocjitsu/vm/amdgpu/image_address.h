// Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_ADDRESS_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_ADDRESS_H_

#include <algorithm>
#include <array>
#include <bit>
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
    return uint64_t{y} * ((uint64_t{width} * bytes + 127u) & ~uint64_t{127}) + uint64_t{x} * bytes;
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

/// GFX11 single-level 2D addressing for GB_ADDR_CONFIG=0x545.
/// XOR equations were checked against AddrLib's Gfx11 surface address API.
inline std::optional<uint64_t> gfx11_image_offset(uint32_t x, uint32_t y, uint32_t width,
                                                  uint32_t bytes, uint32_t swizzle) {
  if (!width || x >= width || (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 16))
    return std::nullopt;
  if (swizzle == 0)
    return uint64_t{y} * ((uint64_t{width} * bytes + 255u) & ~uint64_t{255}) + uint64_t{x} * bytes;
  if (swizzle == 2 || swizzle == 6 || swizzle == 10)
    return gfx12_image_offset(x, y, width, bytes, swizzle ? (swizzle + 2) / 4 : 0);
  if (swizzle == 22) {
    const uint32_t element_log2 = std::countr_zero(bytes);
    const uint32_t x_bits = (13 - element_log2) / 2, y_bits = (12 - element_log2) / 2;
    const uint32_t pipe_xor = (((x >> (x_bits + 1)) & 1u) << 8) |
                              (((y >> (y_bits + 1)) & 1u) << 9) | (((x >> x_bits) & 1u) << 10) |
                              (((y >> y_bits) & 1u) << 11);
    return *gfx12_image_offset(x, y, width, bytes, 2) ^ pipe_xor;
  }
  const bool display = swizzle == 26 || swizzle == 30;
  const bool render = swizzle == 24 || swizzle == 27 || swizzle == 28 || swizzle == 31;
  if ((!display && !render) || ((swizzle == 24 || swizzle == 28) && bytes == 16))
    return std::nullopt;
  const uint32_t block_log2 = swizzle >= 28 ? 18 : 16;
  // Low and high halves select XOR inputs from x and y respectively. Bits
  // above the block dimensions also participate in pipe/bank selection.
  static constexpr uint32_t masks[4][5][18] = {
      {
          // 64 KiB D_X
          {0x00000001, 0x00000002, 0x00010000, 0x00000004, 0x00020000, 0x00040000, 0x00000008,
           0x00080000, 0x00100100, 0x01000010, 0x00200080, 0x00800020, 0x00400040, 0x00000040,
           0x00800000, 0x00000080},
          {0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004, 0x00040000,
           0x00000008, 0x00080100, 0x00800010, 0x00100080, 0x00400020, 0x00200040, 0x00000040,
           0x00400000, 0x00000080},
          {0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004,
           0x00040000, 0x00080080, 0x00800008, 0x00100040, 0x00400010, 0x00200020, 0x00000020,
           0x00400000, 0x00000040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00000004,
           0x00020000, 0x00040080, 0x00400008, 0x00080040, 0x00200010, 0x00100020, 0x00000020,
           0x00200000, 0x00000040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002,
           0x00020000, 0x00040040, 0x00400004, 0x00080020, 0x00200008, 0x00100010, 0x00000010,
           0x00200000, 0x00000020},
      },
      {
          // 256 KiB D_X
          {0x00000001, 0x00000002, 0x00010000, 0x00000004, 0x00020000, 0x00040000, 0x00000008,
           0x00080000, 0x00100100, 0x01000010, 0x00200080, 0x00800020, 0x00400040, 0x00000040,
           0x00800000, 0x00000080, 0x01000000, 0x00000100},
          {0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004, 0x00040000,
           0x00000008, 0x00080100, 0x00800010, 0x00100080, 0x00400020, 0x00200040, 0x00000040,
           0x00400000, 0x00000080, 0x00800000, 0x00000100},
          {0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004,
           0x00040000, 0x00080080, 0x00800008, 0x00100040, 0x00400010, 0x00200020, 0x00000020,
           0x00400000, 0x00000040, 0x00800000, 0x00000080},
          {0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00000004,
           0x00020000, 0x00040080, 0x00400008, 0x00080040, 0x00200010, 0x00100020, 0x00000020,
           0x00200000, 0x00000040, 0x00400000, 0x00000080},
          {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002,
           0x00020000, 0x00040040, 0x00400004, 0x00080020, 0x00200008, 0x00100010, 0x00000010,
           0x00200000, 0x00000020, 0x00400000, 0x00000040},
      },
      {
          // 64 KiB R_X/Z_X
          {0x00000001, 0x00000002, 0x00010000, 0x00000004, 0x00020000, 0x00040000, 0x00000008,
           0x00080000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00000080,
           0x00800000, 0x00800040},
          {0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004, 0x00040000,
           0x00000008, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00080000,
           0x00000080, 0x00800040},
          {0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004,
           0x00040000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00000008,
           0x00080000, 0x00800040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00000004,
           0x00020000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00440080, 0x00000008,
           0x00080000, 0x00800040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002,
           0x00020000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00440080, 0x00000008,
           0x00080000, 0x00800044},
      },
      {
          // 256 KiB R_X/Z_X
          {0x00000001, 0x00000002, 0x00010000, 0x00000004, 0x00020000, 0x00040000, 0x00000008,
           0x00080000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00000080,
           0x00800000, 0x02000100, 0x01000200, 0x00800040},
          {0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004, 0x00040000,
           0x00000008, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00080000,
           0x00000080, 0x01000100, 0x00800200, 0x00800040},
          {0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00020000, 0x00000004,
           0x00040000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00000008,
           0x00080000, 0x01000080, 0x00800100, 0x00800040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002, 0x00000004,
           0x00020000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00040000,
           0x00000008, 0x00080100, 0x00800080, 0x00800040},
          {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000002,
           0x00020000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00000004,
           0x00040000, 0x00080080, 0x00800008, 0x00800040},
      },
  };
  const uint32_t element_log2 = std::countr_zero(bytes);
  const uint32_t pattern = (render ? 2 : 0) + (block_log2 == 18);
  uint32_t offset = 0;
  for (uint32_t bit = 0; bit < block_log2; ++bit) {
    const uint32_t mask = masks[pattern][element_log2][bit];
    offset |= ((std::popcount(x & (mask & 0xffff)) + std::popcount(y & (mask >> 16))) & 1u) << bit;
  }
  const uint32_t x_bits = (block_log2 - element_log2 + 1) / 2;
  const uint32_t y_bits = (block_log2 - element_log2) / 2;
  const uint32_t pitch_blocks = (width + (1u << x_bits) - 1) >> x_bits;
  const uint64_t block = uint64_t{y >> y_bits} * pitch_blocks + (x >> x_bits);
  return (block << block_log2) + offset;
}

inline std::optional<uint64_t> gfx11_image_address(uint64_t base, uint32_t x, uint32_t y,
                                                   uint32_t width, uint32_t bytes,
                                                   uint32_t swizzle) {
  const auto offset = gfx11_image_offset(x, y, width, bytes, swizzle);
  if (!offset)
    return std::nullopt;
  if (!swizzle)
    return base + *offset;
  const uint32_t block_log2 = swizzle == 2                      ? 8
                              : (swizzle == 6 || swizzle == 22) ? 12
                              : swizzle >= 28                   ? 18
                                                                : 16;
  const uint64_t block_mask = (uint64_t{1} << block_log2) - 1;
  return (base & ~block_mask) + (*offset ^ (base & block_mask));
}

} // namespace rocjitsu::amdgpu

#endif
