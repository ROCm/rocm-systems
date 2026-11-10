// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GRAPHICS_IMAGE_VOLUME_H_
#define ROCJITSU_VM_GRAPHICS_IMAGE_VOLUME_H_

#include "rocjitsu/vm/graphics/image_address.h"

namespace rocjitsu::amdgpu {

// Three-dimensional tile equations, evaluated from four-bit coordinate chunks.
struct ImageVolumeEquation {
  std::array<std::array<std::array<uint32_t, 16>, 2>, 3> chunks{};
  std::array<uint32_t, 3> extent{1, 1, 1};
  uint32_t offset(uint32_t x, uint32_t y, uint32_t z) const {
    return chunks[0][0][x & 15] ^ chunks[0][1][(x >> 4) & 15] ^ chunks[1][0][y & 15] ^
           chunks[1][1][(y >> 4) & 15] ^ chunks[2][0][z & 15] ^ chunks[2][1][(z >> 4) & 15];
  }
};

inline const ImageVolumeEquation &gfx12_volume_equation(uint32_t swizzle, uint32_t bytes) {
  // AddrLib gfx12SwizzlePattern.h, 4KB/64KB/256KB 3D, one sample.
  static constexpr std::array<std::array<uint8_t, 18>, 5> patterns{{
      {{1, 2, 33, 17, 18, 34, 3, 35, 19, 4, 36, 20, 5, 37, 21, 6, 38, 22}},
      {{0, 1, 33, 17, 2, 34, 18, 35, 19, 3, 36, 20, 4, 37, 21, 5, 38, 22}},
      {{0, 0, 1, 17, 2, 33, 18, 34, 19, 3, 35, 20, 4, 36, 21, 5, 37, 22}},
      {{0, 0, 0, 1, 17, 33, 2, 34, 18, 3, 35, 19, 4, 36, 20, 5, 37, 21}},
      {{0, 0, 0, 0, 1, 33, 17, 34, 18, 2, 35, 19, 3, 36, 20, 4, 37, 21}},
  }};
  static constexpr auto equations = [] {
    std::array<std::array<ImageVolumeEquation, 5>, 3> result{};
    for (uint32_t mode = 0; mode < 3; ++mode)
      for (uint32_t element = 0; element < 5; ++element) {
        auto &equation = result[mode][element];
        const uint32_t bits = mode == 2 ? 18 : 12 + 4 * mode;
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const uint32_t selector = patterns[element][bit];
          if (!selector)
            continue;
          const uint32_t axis = (selector - 1) / 16, input = (selector - 1) % 16;
          equation.extent[axis] = std::max(equation.extent[axis], 2u << input);
          for (uint32_t value = 0; value < 16; ++value)
            equation.chunks[axis][input / 4][value] |= ((value >> (input % 4)) & 1u) << bit;
        }
      }
    return result;
  }();
  return equations.at(swizzle - 5).at(std::countr_zero(bytes));
}

// GFX11 standard 3D tiles use the same coordinate ordering as GFX12, with
// pipe/bank XOR applied to bits 8..12 for GB_ADDR_CONFIG=0x545.
inline const ImageVolumeEquation &gfx11_volume_equation(uint32_t swizzle, uint32_t bytes) {
  if (swizzle == 5 || swizzle == 9)
    return gfx12_volume_equation(swizzle == 5 ? 5 : 6, bytes);
  static const auto equations = [] {
    // S_T, 4KB_S_X, 64KB_S_X and 256KB_S_X. Each mask packs X/Y/Z
    // inputs in successive 16-bit fields, before scaling for element size.
    std::array<std::array<ImageVolumeEquation, 5>, 4> result{};
    constexpr uint64_t extra[] = {0x008000000080, 0x004000400000, 0x000000200040, 0x002000000020,
                                  0x001000100000};
    for (uint32_t mode = 0; mode < 4; ++mode)
      for (uint32_t element = 0; element < 5; ++element) {
        auto &equation = result[mode][element];
        equation = gfx12_volume_equation(mode == 1 ? 5 : mode == 3 ? 7 : 6, 1u << element);
        const uint32_t shifts[] = {(element + 2) / 3, element / 3, (element + 1) / 3};
        for (uint32_t bit = 0; bit < (mode == 1 ? 4u : 5u); ++bit)
          for (uint32_t axis = 0; axis < 3; ++axis) {
            uint32_t mask = ((extra[bit] >> (axis * 16)) & 0xffffu) >> (shifts[axis] + (mode == 1));
            // S_T excludes coordinates outside the macroblock.
            if (mode == 0)
              mask &= equation.extent[axis] - 1;
            for (uint32_t chunk = 0; chunk < 2; ++chunk)
              for (uint32_t value = 0; value < 16; ++value)
                equation.chunks[axis][chunk][value] ^=
                    (std::popcount(value & (mask >> (chunk * 4))) & 1u) << (bit + 8);
          }
      }
    return result;
  }();
  return equations.at((swizzle - 17) / 4).at(std::countr_zero(bytes));
}

inline bool image_volume_thick(bool gfx12, uint32_t swizzle) {
  return gfx12 ? swizzle >= 5
               : swizzle == 5 || swizzle == 9 || swizzle == 17 || swizzle == 21 || swizzle == 25 ||
                     swizzle == 29;
}

struct ImageVolumeMipLayout {
  ImageMipLayout plane;
  uint32_t depth = 1;
};

inline std::optional<ImageVolumeMipLayout>
image_volume_mip_layout(bool gfx12, uint32_t swizzle, uint32_t bytes, uint32_t width,
                        uint32_t height, uint32_t depth, uint32_t levels, uint32_t level) {
  if (!width || !height || !depth || width > 65536 || height > 65536 || depth > 16384 || !levels ||
      levels > 17 || level >= levels || !std::has_single_bit(bytes) || bytes > 16 ||
      (gfx12 ? swizzle > 7
             : !image_volume_thick(false, swizzle) && swizzle != 0 && swizzle != 24 &&
                   swizzle != 27 && swizzle != 28 && swizzle != 31))
    return std::nullopt;
  ImageVolumeMipLayout result{{}, std::max(1u, depth >> level)};
  auto &out = result.plane;
  if (!image_volume_thick(gfx12, swizzle)) {
    const auto plane = image_mip_layout(gfx12, swizzle, bytes, width, height, levels, level);
    if (!plane)
      return std::nullopt;
    out = *plane;
    return result;
  }
  const auto &equation =
      gfx12 ? gfx12_volume_equation(swizzle, bytes) : gfx11_volume_equation(swizzle, bytes);
  const auto [bw, bh, bd] = equation.extent;
  const uint32_t bits = std::countr_zero(bw * bh * bd * bytes);
  const uint32_t effective_bits = bits - (bits - 8) / 3;
  const uint32_t max_tail =
      effective_bits <= 11 ? 1 + (1u << (effective_bits - 9)) : effective_bits - 4;
  const auto align = [](uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  const auto extent = [](uint32_t value, uint32_t mip) { return (value + (1u << mip) - 1) >> mip; };
  std::array<uint32_t, 3> tail = equation.extent;
  tail[bits % 3 == 0 ? 1 : bits % 3 == 1 ? 0 : 2] /= 2;
  uint32_t first_tail = levels;
  // Tail admission uses X/Y extent; depth can span additional volume tiles.
  if (levels > 1)
    for (uint32_t i = 0; i < levels; ++i)
      if (extent(width, i) <= tail[0] && extent(height, i) <= tail[1] && levels - i <= max_tail) {
        first_tail = i;
        break;
      }
  out.width = std::max(1u, width >> level);
  out.height = std::max(1u, height >> level);
  out.pitch = align(extent(width, level), bw);
  out.first_tail = first_tail;
  out.slice_size = first_tail < levels ? (1u << bits) / bd : 0;
  for (uint32_t i = 0; i < first_tail; ++i) {
    const uint64_t slice =
        uint64_t{align(extent(width, i), bw)} * align(extent(height, i), bh) * bytes;
    out.slice_size += slice;
    if (i > level)
      out.offset += slice * bd;
  }
  if (level < first_tail) {
    if (first_tail < levels)
      out.offset += 1u << bits;
  } else {
    const uint32_t slot = max_tail - 1 - (level - first_tail);
    const uint32_t offset = slot > 6 ? 16u << slot : slot << 8;
    const uint32_t micro_bits = 8 - std::countr_zero(bytes);
    const uint32_t micro_width = 1u << (micro_bits / 3 + (micro_bits % 3 > 1));
    const uint32_t micro_height = 1u << (micro_bits / 3);
    for (uint32_t bit = 0; bit < 6; ++bit) {
      out.tail_x |= ((offset >> (9 + 2 * bit)) & 1u) << bit;
      out.tail_y |= ((offset >> (8 + 2 * bit)) & 1u) << bit;
    }
    out.tail_x *= micro_width;
    out.tail_y *= micro_height;
    out.pitch = bw;
  }
  return result;
}

inline std::optional<uint64_t> image_volume_address(bool gfx12, uint64_t base,
                                                    const ImageVolumeMipLayout &mip, uint32_t x,
                                                    uint32_t y, uint32_t z, uint32_t bytes,
                                                    uint32_t swizzle) {
  const auto &plane = mip.plane;
  if (x >= plane.width || y >= plane.height || z >= mip.depth)
    return std::nullopt;
  if (!image_volume_thick(gfx12, swizzle)) {
    const uint64_t slice_base =
        image_layer_base(gfx12, base + plane.offset, plane.slice_size, z, bytes, swizzle);
    return gfx12 ? gfx12_image_address(slice_base, x + plane.tail_x, y + plane.tail_y, plane.pitch,
                                       bytes, swizzle)
                 : gfx11_image_address(slice_base, x + plane.tail_x, y + plane.tail_y, plane.pitch,
                                       bytes, swizzle);
  }
  const auto &equation =
      gfx12 ? gfx12_volume_equation(swizzle, bytes) : gfx11_volume_equation(swizzle, bytes);
  const auto [bw, bh, bd] = equation.extent;
  const uint64_t block = uint64_t{y / bh} * (plane.pitch / bw) + x / bw;
  const uint64_t block_bytes = uint64_t{bw} * bh * bd * bytes;
  const uint64_t offset = plane.offset + uint64_t{z / bd} * plane.slice_size * bd +
                          block * block_bytes +
                          equation.offset(x + plane.tail_x, y + plane.tail_y, z);
  // The descriptor's low address bits encode pipe/bank XOR within each tile.
  return (base & ~(block_bytes - 1)) + (offset ^ (base & (block_bytes - 1)));
}

} // namespace rocjitsu::amdgpu

#endif
