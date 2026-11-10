// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/graphics/image_address.h"
#include "util/result.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::amdgpu {

/// Single-sample GFX11 metadata addressing for GB_ADDR_CONFIG=0x545.
/// These XOR equations and block dimensions follow AddrLib's GFX11 metadata API.
inline std::optional<uint64_t> gfx11_metadata_address(uint64_t base, uint32_t x, uint32_t y,
                                                      uint32_t width, uint32_t height,
                                                      uint32_t bytes, uint32_t swizzle, bool depth,
                                                      bool pipe_aligned = true,
                                                      uint32_t layer = 0) {
  if (!width || !height || x >= width || y >= height || !std::has_single_bit(bytes) || bytes > 16 ||
      (depth && bytes != 1 && bytes != 2 && bytes != 4) ||
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
  static constexpr auto htile_equation = make_image_equation(htile_masks, 17);
  static constexpr auto dcc_equations = []() consteval {
    std::array<std::array<ImageAddressEquation, 5>, 2> result{};
    for (uint32_t mode = 0; mode < 2; ++mode)
      for (uint32_t element = 0; element < 5; ++element)
        result[mode][element] = make_image_equation(dcc_masks[mode][element], 14);
    return result;
  }();
  static constexpr auto unaligned_equations = []() consteval {
    std::array<ImageAddressEquation, 5> result{};
    for (uint32_t element = 0; element < 5; ++element) {
      uint32_t masks[12]{};
      for (uint32_t bit = 0; bit < 12; ++bit) {
        const uint32_t dimension = (bit + element) & 1;
        const uint32_t coordinate = (bit + 8 - element) / 2;
        masks[bit] = 1u << (coordinate + (dimension ? 16 : 0));
      }
      result[element] = make_image_equation(masks, 12);
    }
    return result;
  }();
  const auto &equation = depth          ? htile_equation
                         : pipe_aligned ? dcc_equations[swizzle == 31][element_log2]
                                        : unaligned_equations[element_log2];
  uint32_t offset = equation.offset(x, y);
  const uint64_t pitch_blocks = (uint64_t{width} + (1u << xb) - 1) >> xb;
  const uint64_t slice_blocks = pitch_blocks * ((uint64_t{height} + (1u << yb) - 1) >> yb);
  const uint64_t block =
      uint64_t{layer} * slice_blocks + uint64_t{y >> yb} * pitch_blocks + (x >> xb);
  const uint64_t mask = (1u << block_log2) - 1;
  if (depth || pipe_aligned)
    offset ^= gfx11_image_slice_xor(layer, bytes, swizzle) & mask;
  return (base & ~mask) + (block << block_log2) + (offset ^ (base & mask));
}

/// GFX11 DCC allocates a metadata block for the first packed mip and then
/// whole metadata blocks for each larger mip, in reverse level order. Later
/// tail levels cannot use DCC (PAL Image::CanMipSupportMetaData).
struct Gfx11DccMipLayout {
  ImageMipLayout pixels;
  uint64_t offset = 0, slice_size = 0;
  uint32_t width = 0, height = 0;
  bool enabled = false;
};

inline std::optional<Gfx11DccMipLayout> gfx11_dcc_mip_layout(uint32_t swizzle, uint32_t bytes,
                                                             uint32_t width, uint32_t height,
                                                             uint32_t levels, uint32_t level,
                                                             bool pipe_aligned) {
  if (swizzle != 27 && swizzle != 31)
    return std::nullopt;
  const auto pixels = image_mip_layout(false, swizzle, bytes, width, height, levels, level);
  if (!pixels)
    return std::nullopt;
  const uint32_t block_log2 = pipe_aligned ? 14 : 12;
  const uint32_t bits = block_log2 + 8 - std::countr_zero(bytes);
  const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
  const auto extent = [](uint32_t size, uint32_t mip) { return (size + (1u << mip) - 1) >> mip; };
  Gfx11DccMipLayout result{.pixels = *pixels,
                           .width = extent(width, level),
                           .height = extent(height, level),
                           .enabled = level <= pixels->first_tail};
  result.slice_size = pixels->first_tail < levels ? uint64_t{1} << block_log2 : 0;
  for (uint32_t mip = pixels->first_tail; mip-- > 0;) {
    if (mip == level)
      result.offset = result.slice_size;
    result.slice_size +=
        uint64_t{(extent(width, mip) + bw - 1) / bw} * ((extent(height, mip) + bh - 1) / bh)
        << block_log2;
  }
  return result;
}

/// A failed result denotes an unsupported metadata layout or clear key.
/// VM outcomes retain their read/write retry and fault semantics.
using ImageMetadataResult = util::FailureOr<VmAccessOutcome>;

/// Materialize one DCC clear block, retaining the uncompressed metadata
/// encoding. General delta compression is never produced by the functional
/// renderer.
template <typename Memory>
ImageMetadataResult
materialize_gfx11_dcc_at_address(const Memory &memory, uint64_t base, uint64_t address, uint32_t x,
                                 uint32_t y, uint32_t width, uint32_t height, uint32_t bytes,
                                 uint32_t swizzle, uint32_t layer = 0, uint64_t slice_size = 0,
                                 uint32_t pitch = 0, uint32_t tail_x = 0, uint32_t tail_y = 0) {
  if (!pitch)
    pitch = width;
  base = image_layer_base(false, base, slice_size, layer, bytes, swizzle);
  uint8_t key;
  if (const VmAccessOutcome outcome =
          memory.read(address, std::as_writable_bytes(std::span{&key, 1}));
      outcome != VmAccessOutcome::Complete)
    return outcome;
  if (key == 0xff)
    return VmAccessOutcome::Complete;
  const uint32_t bits = 8 - std::countr_zero(bytes);
  const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
  x &= ~(bw - 1);
  y &= ~(bh - 1);
  std::array<uint8_t, 16> value{};
  if (key == 1) {
    const auto clear = gfx11_image_address(base, x + tail_x, y + tail_y, pitch, bytes, swizzle);
    if (!clear)
      return util::Result::failure();
    if (const VmAccessOutcome outcome =
            memory.read(*clear, std::as_writable_bytes(std::span{value}.first(bytes)));
        outcome != VmAccessOutcome::Complete)
      return outcome;
  } else if (key == 2) {
    value.fill(0xff);
  } else if (key == 4 || key == 6) {
    const uint32_t step = key == 4 ? 2 : 4;
    const uint32_t one = key == 4 ? 0x3c00 : 0x3f800000;
    if (bytes % step)
      return util::Result::failure();
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = one >> (8 * (i % step));
  } else if (key == 8 || key == 10) {
    if (bytes != 2 && bytes != 4 && bytes != 8)
      return util::Result::failure();
    const uint32_t last_component = bytes == 2 ? 1 : 3 * bytes / 4;
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = ((i >= last_component) == (key == 8)) ? 0xff : 0;
  } else if (key != 0) {
    return util::Result::failure();
  }
  for (uint32_t py = y; py < std::min(y + bh, height); ++py)
    for (uint32_t px = x; px < std::min(x + bw, width); ++px)
      if (const VmAccessOutcome outcome = memory.write(
              *gfx11_image_address(base, px + tail_x, py + tail_y, pitch, bytes, swizzle),
              std::as_bytes(std::span{value}.first(bytes)));
          outcome != VmAccessOutcome::Complete)
        return outcome;
  key = 0xff;
  if (const VmAccessOutcome outcome = memory.write(address, std::as_bytes(std::span{&key, 1}));
      outcome != VmAccessOutcome::Complete)
    return outcome;
  return VmAccessOutcome::Complete;
}

template <typename Memory>
ImageMetadataResult
materialize_gfx11_dcc(const Memory &memory, uint64_t base, uint64_t metadata, uint32_t x,
                      uint32_t y, uint32_t width, uint32_t height, uint32_t bytes, uint32_t swizzle,
                      bool pipe_aligned = true, uint32_t layer = 0, uint64_t slice_size = 0) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, false,
                                              pipe_aligned, layer);
  if (!address)
    return util::Result::failure();
  return materialize_gfx11_dcc_at_address(memory, base, *address, x, y, width, height, bytes,
                                          swizzle, layer, slice_size);
}

/// Materialize a selected mip without touching adjacent levels or array slices.
template <typename Memory>
ImageMetadataResult materialize_gfx11_dcc_mip(const Memory &memory, uint64_t base,
                                              uint64_t metadata, uint32_t x, uint32_t y,
                                              uint32_t bytes, uint32_t swizzle, bool pipe_aligned,
                                              uint32_t layer, const Gfx11DccMipLayout &mip) {
  if (!mip.enabled)
    return VmAccessOutcome::Complete;
  const uint32_t metadata_mask = (1u << (pipe_aligned ? 14 : 12)) - 1;
  metadata += mip.offset + layer * mip.slice_size;
  if (pipe_aligned)
    metadata ^= gfx11_image_slice_xor(layer, bytes, swizzle) & metadata_mask;
  const auto address = gfx11_metadata_address(metadata, x, y, mip.width, mip.height, bytes, swizzle,
                                              false, pipe_aligned);
  if (!address)
    return util::Result::failure();
  const auto &pixels = mip.pixels;
  return materialize_gfx11_dcc_at_address(
      memory, base + pixels.offset, *address, x, y, pixels.width, pixels.height, bytes, swizzle,
      layer, pixels.slice_size, pixels.pitch, pixels.tail_x, pixels.tail_y);
}

/// HTILE ZMask zero references DB_DEPTH_CLEAR; ZMask fifteen is uncompressed.
template <typename Memory>
ImageMetadataResult materialize_gfx11_htile_at_address(
    const Memory &memory, uint64_t base, uint64_t address, uint32_t x, uint32_t y, uint32_t width,
    uint32_t height, uint32_t bytes, uint32_t swizzle,
    std::optional<uint32_t> clear_bits = std::nullopt, bool has_stencil = false) {
  if (bytes != 1 && bytes != 2 && bytes != 4)
    return util::Result::failure();
  uint32_t key;
  if (const VmAccessOutcome outcome =
          memory.read(address, std::as_writable_bytes(std::span{&key, 1}));
      outcome != VmAccessOutcome::Complete)
    return outcome;
  if (bytes == 1) {
    if ((key & 0x300u) != 0x300u)
      return util::Result::failure();
    return VmAccessOutcome::Complete;
  }
  if ((key & 15) == 15)
    return VmAccessOutcome::Complete;
  if (key & 15)
    return util::Result::failure();
  if (!clear_bits) {
    // Texture-compatible fast clears encode the endpoints exactly. Other
    // clears require DB_DEPTH_CLEAR, which an image descriptor cannot supply.
    if (key == 0)
      clear_bits = 0;
    else if (key == 0xfffffff0)
      clear_bits = bytes == 2 ? 65535 : 0x3f800000;
    else
      return util::Result::failure();
  }
  x &= ~7u;
  y &= ~7u;
  for (uint32_t py = y; py < std::min(y + 8, height); ++py)
    for (uint32_t px = x; px < std::min(x + 8, width); ++px)
      if (const VmAccessOutcome outcome =
              memory.write(*gfx11_image_address(base, px, py, width, bytes, swizzle),
                           std::as_bytes(std::span{&*clear_bits, 1}).first(bytes));
          outcome != VmAccessOutcome::Complete)
        return outcome;
  key = has_stencil ? (key & 0x3f0u) | 0xfffff00fu : 0xfffc000fu;
  if (const VmAccessOutcome outcome = memory.write(address, std::as_bytes(std::span{&key, 1}));
      outcome != VmAccessOutcome::Complete)
    return outcome;
  return VmAccessOutcome::Complete;
}

template <typename Memory>
ImageMetadataResult materialize_gfx11_htile(const Memory &memory, uint64_t base, uint64_t metadata,
                                            uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                            uint32_t bytes, uint32_t swizzle,
                                            std::optional<uint32_t> clear_bits = std::nullopt,
                                            bool has_stencil = false) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, true);
  if (!address)
    return util::Result::failure();
  return materialize_gfx11_htile_at_address(memory, base, *address, x, y, width, height, bytes,
                                            swizzle, clear_bits, has_stencil);
}

} // namespace rocjitsu::amdgpu
