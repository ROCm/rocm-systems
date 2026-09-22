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

/// Block-size log2 for a validated swizzle mode; zero denotes linear storage.
inline uint32_t image_block_log2(bool gfx12, uint32_t swizzle) {
  if (!swizzle)
    return 0;
  if (gfx12)
    return swizzle == 4 ? 18 : 4 * swizzle + 4;
  if (swizzle == 2)
    return 8;
  if (swizzle == 6 || swizzle == 22)
    return 12;
  return swizzle >= 28 ? 18 : 16;
}

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
  const uint32_t block_log2 = image_block_log2(true, swizzle);
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
  const uint32_t block_log2 = image_block_log2(true, swizzle);
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
  const uint32_t block_log2 = image_block_log2(false, swizzle);
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
  const uint32_t block_log2 = image_block_log2(false, swizzle);
  const uint64_t block_mask = (uint64_t{1} << block_log2) - 1;
  return (base & ~block_mask) + (*offset ^ (base & block_mask));
}

/// Accessible mip extents, allocation offset and packed-tail origin.
struct ImageMipLayout {
  uint64_t offset = 0;
  uint32_t width = 0, height = 0, pitch = 0;
  uint32_t tail_x = 0, tail_y = 0;
};

/// Single-sample 2D mip allocation for the layouts supported above. The offset
/// is block aligned; tail coordinates must pass through the swizzle equation.
inline std::optional<ImageMipLayout> image_mip_layout(bool gfx12, uint32_t swizzle, uint32_t bytes,
                                                      uint32_t width, uint32_t height,
                                                      uint32_t levels, uint32_t level) {
  if (!width || !height || width > 65536 || height > 65536 || !levels || level >= levels ||
      levels > uint32_t(std::bit_width(std::max(width, height))) ||
      !(gfx12 ? gfx12_image_offset(0, 0, width, bytes, swizzle)
              : gfx11_image_offset(0, 0, width, bytes, swizzle)))
    return std::nullopt;
  const uint32_t element_log2 = std::countr_zero(bytes);
  const uint32_t block_log2 = image_block_log2(gfx12, swizzle);
  const uint32_t block_width = swizzle ? 1u << ((block_log2 - element_log2 + 1) / 2) : 256 / bytes;
  const uint32_t block_height = swizzle ? 1u << ((block_log2 - element_log2) / 2) : 1;
  const auto align = [](uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  // Storage extents round up at odd sizes; accessible texel extents round down.
  const auto allocation_extent = [](uint32_t value, uint32_t mip) {
    return (value + (1u << mip) - 1) >> mip;
  };
  ImageMipLayout out{.width = std::max(1u, width >> level),
                     .height = std::max(1u, height >> level),
                     .pitch = width};
  if (levels == 1)
    return out;
  out.pitch = align(allocation_extent(width, level), block_width);
  const uint32_t micro_width = 1u << ((8 - element_log2 + 1) / 2);
  const uint32_t micro_height = 1u << ((8 - element_log2) / 2);
  uint32_t tail_width = block_width / 2, tail_height = block_height;
  // GFX11 Z swizzles use the 32-bit depth threshold for smaller elements too.
  if (!gfx12 && (swizzle == 24 || swizzle == 28) && bytes < 4) {
    tail_width /= micro_width / 8;
    tail_height /= micro_height / 8;
  }
  const uint32_t max_tail_levels = block_log2 > 8 ? block_log2 - 4 : 0;
  uint32_t first_tail = levels;
  for (uint32_t mip = 0; mip < levels; ++mip) {
    if (allocation_extent(width, mip) <= tail_width &&
        allocation_extent(height, mip) <= tail_height && levels - mip <= max_tail_levels) {
      first_tail = mip;
      break;
    }
  }
  if (level >= first_tail) {
    const uint32_t slot = max_tail_levels - 1 - (level - first_tail);
    const uint32_t tail = slot > 6 ? 16u << slot : slot << 8;
    // Tail slots describe a grid of 256-byte microtiles, before pipe/bank XOR.
    for (uint32_t bit = 0; bit < 6; ++bit) {
      out.tail_x |= ((tail >> (9 + 2 * bit)) & 1u) << bit;
      out.tail_y |= ((tail >> (8 + 2 * bit)) & 1u) << bit;
    }
    out.tail_x *= micro_width;
    out.tail_y *= micro_height;
    out.pitch = block_width;
  } else {
    // Small mips precede large mips; a packed tail occupies one whole block.
    if (first_tail < levels)
      out.offset = uint64_t{1} << block_log2;
    for (uint32_t mip = level + 1; mip < first_tail; ++mip)
      out.offset += uint64_t{align(allocation_extent(width, mip), block_width)} *
                    align(allocation_extent(height, mip), block_height) * bytes;
  }
  // GFX12 linear storage still allocates 256-byte rows, but addresses pixels
  // with a 128-byte row pitch.
  if (gfx12 && !swizzle)
    out.pitch = align(allocation_extent(width, level), 128 / bytes);
  return out;
}

} // namespace rocjitsu::amdgpu

#endif
