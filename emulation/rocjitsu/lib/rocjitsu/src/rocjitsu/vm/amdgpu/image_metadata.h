// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_METADATA_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_METADATA_H_

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/image_address.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace rocjitsu::amdgpu {

/// Single-sample GFX11 metadata addressing for GB_ADDR_CONFIG=0x545.
/// These XOR equations and block dimensions follow AddrLib's GFX11 metadata API.
inline std::optional<uint64_t> gfx11_metadata_address(uint64_t base, uint32_t x, uint32_t y,
                                                      uint32_t width, uint32_t height,
                                                      uint32_t bytes, uint32_t swizzle, bool depth,
                                                      bool pipe_aligned = true,
                                                      uint32_t layer = 0) {
  if (!width || !height || x >= width || y >= height || !std::has_single_bit(bytes) || bytes > 16 ||
      (depth && bytes != 2 && bytes != 4) ||
      (depth ? (swizzle != 24 && swizzle != 28) : (swizzle != 27 && swizzle != 31)))
    return std::nullopt;
  const uint32_t element_log2 = std::countr_zero(bytes);
  static constexpr uint32_t dcc_masks[2][5][14] = {
      {
          {0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200, 0x02000000, 0x00000400,
           0x04000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200, 0x02000000,
           0x00000400, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200,
           0x02000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00040000,
           0x00000200, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00440080, 0x00800040},
          {0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000004,
           0x00040000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00440080, 0x00800040},
      },
      {
          {0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200, 0x02000000, 0x00000400,
           0x04000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200, 0x02000000,
           0x00000400, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000, 0x00000200,
           0x02000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00040000, 0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100, 0x01000000,
           0x00000200, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
          {0x00000004, 0x00040000, 0x00000008, 0x00080000, 0x00000080, 0x00800000, 0x00000100,
           0x01000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020, 0x00400080, 0x00800040},
      },
  };
  static constexpr uint32_t htile_masks[] = {
      0x00000000, 0x00000000, 0x00000008, 0x00080000, 0x00000080, 0x00800000,
      0x00000100, 0x01000000, 0x02100200, 0x00100010, 0x00200100, 0x01000020,
      0x00400080, 0x00000200, 0x02000000, 0x00000400, 0x00800040};
  const uint32_t block_log2 = depth ? 17 : pipe_aligned ? 14 : 12;
  const uint32_t pixel_bits = depth ? 21 : block_log2 + 8 - element_log2;
  const uint32_t xb = (pixel_bits + 1) / 2, yb = pixel_bits / 2;
  uint32_t offset = 0;
  for (uint32_t bit = 0; bit < block_log2; ++bit) {
    uint32_t mask;
    if (depth) {
      mask = htile_masks[bit];
    } else if (pipe_aligned) {
      mask = dcc_masks[swizzle == 31][element_log2][bit];
    } else {
      const uint32_t dimension = (bit + element_log2) & 1;
      const uint32_t coordinate = (bit + 8 - element_log2) / 2;
      mask = 1u << (coordinate + (dimension ? 16 : 0));
    }
    offset |= ((std::popcount(x & (mask & 0xffff)) + std::popcount(y & (mask >> 16))) & 1u) << bit;
  }
  const uint64_t pitch_blocks = (uint64_t{width} + (1u << xb) - 1) >> xb;
  const uint64_t slice_blocks = pitch_blocks * ((uint64_t{height} + (1u << yb) - 1) >> yb);
  const uint64_t block =
      uint64_t{layer} * slice_blocks + uint64_t{y >> yb} * pitch_blocks + (x >> xb);
  const uint64_t mask = (1u << block_log2) - 1;
  if (depth || pipe_aligned)
    offset ^= gfx11_image_slice_xor(layer, bytes, swizzle) & mask;
  return (base & ~mask) + (block << block_log2) + (offset ^ (base & mask));
}

inline void read_image_bytes(const GpuVmAccess &memory, uint64_t address,
                             std::span<uint8_t> bytes) {
  if (memory.read(address, std::as_writable_bytes(bytes)) != VmAccessOutcome::Complete)
    throw std::runtime_error("image read failed");
}

inline void write_image_bytes(const GpuVmAccess &memory, uint64_t address,
                              std::span<const uint8_t> bytes) {
  if (memory.write(address, std::as_bytes(bytes)) != VmAccessOutcome::Complete)
    throw std::runtime_error("image write failed");
}

/// Materialize one DCC clear block, retaining the uncompressed metadata encoding.
/// General delta compression is never produced by the functional renderer.
inline void materialize_gfx11_dcc(const GpuVmAccess &memory, uint64_t base, uint64_t metadata,
                                  uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                  uint32_t bytes, uint32_t swizzle, bool pipe_aligned = true,
                                  uint32_t layer = 0, uint64_t slice_size = 0) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, false,
                                              pipe_aligned, layer);
  if (!address)
    throw std::runtime_error("unsupported GFX11 DCC surface layout");
  base = image_layer_base(false, base, slice_size, layer, bytes, swizzle);
  uint8_t key;
  read_image_bytes(memory, *address, {&key, 1});
  if (key == 0xff)
    return;
  const uint32_t bits = 8 - std::countr_zero(bytes);
  const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
  x &= ~(bw - 1);
  y &= ~(bh - 1);
  std::array<uint8_t, 16> value{};
  if (key == 1) {
    const auto clear = gfx11_image_address(base, x, y, width, bytes, swizzle);
    if (!clear)
      throw std::runtime_error("unsupported GFX11 DCC clear layout");
    read_image_bytes(memory, *clear, {value.data(), bytes});
  } else if (key == 2) {
    value.fill(0xff);
  } else if (key == 4 || key == 6) {
    const uint32_t step = key == 4 ? 2 : 4;
    const uint32_t one = key == 4 ? 0x3c00 : 0x3f800000;
    if (bytes % step)
      throw std::runtime_error("unsupported GFX11 DCC floating clear format");
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = one >> (8 * (i % step));
  } else if (key == 8 || key == 10) {
    if (bytes != 2 && bytes != 4 && bytes != 8)
      throw std::runtime_error("unsupported GFX11 DCC mixed clear format");
    const uint32_t last_component = bytes == 2 ? 1 : 3 * bytes / 4;
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = ((i >= last_component) == (key == 8)) ? 0xff : 0;
  } else if (key != 0) {
    throw std::runtime_error("unsupported GFX11 DCC compressed block");
  }
  for (uint32_t py = y; py < std::min(y + bh, height); ++py)
    for (uint32_t px = x; px < std::min(x + bw, width); ++px)
      write_image_bytes(memory, *gfx11_image_address(base, px, py, width, bytes, swizzle),
                        {value.data(), bytes});
  key = 0xff;
  write_image_bytes(memory, *address, {&key, 1});
}

/// HTILE ZMask zero references DB_DEPTH_CLEAR; ZMask fifteen is uncompressed.
inline void materialize_gfx11_htile(const GpuVmAccess &memory, uint64_t base, uint64_t metadata,
                                    uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                    uint32_t bytes, uint32_t swizzle,
                                    std::optional<uint32_t> clear_bits = std::nullopt) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, true);
  if (!address)
    throw std::runtime_error("unsupported GFX11 HTILE surface layout");
  uint32_t key;
  read_image_bytes(memory, *address, {reinterpret_cast<uint8_t *>(&key), 4});
  if ((key & 15) == 15)
    return;
  if (key & 15)
    throw std::runtime_error("unsupported GFX11 HTILE compressed block");
  if (!clear_bits) {
    // Texture-compatible fast clears encode the endpoints exactly. Other
    // clears require DB_DEPTH_CLEAR, which an image descriptor cannot supply.
    if (key == 0)
      clear_bits = 0;
    else if (key == 0xfffffff0)
      clear_bits = bytes == 2 ? 65535 : 0x3f800000;
    else
      throw std::runtime_error("GFX11 HTILE clear requires a depth clear register");
  }
  x &= ~7u;
  y &= ~7u;
  for (uint32_t py = y; py < std::min(y + 8, height); ++py)
    for (uint32_t px = x; px < std::min(x + 8, width); ++px)
      write_image_bytes(memory, *gfx11_image_address(base, px, py, width, bytes, swizzle),
                        {reinterpret_cast<const uint8_t *>(&*clear_bits), bytes});
  key = 0xfffc000f;
  write_image_bytes(memory, *address, {reinterpret_cast<const uint8_t *>(&key), 4});
}

} // namespace rocjitsu::amdgpu
#endif
