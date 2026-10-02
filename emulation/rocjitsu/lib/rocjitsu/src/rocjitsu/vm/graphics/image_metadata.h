// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GRAPHICS_IMAGE_METADATA_H_
#define ROCJITSU_VM_GRAPHICS_IMAGE_METADATA_H_

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/graphics/image_address.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu::amdgpu {

/// Padded GFX11 metadata envelope, including all bytes permuted by block XOR.
inline std::optional<VmRamRange> gfx11_metadata_range(uint64_t base, uint32_t width,
                                                      uint32_t height, uint32_t bytes, bool depth,
                                                      bool pipe_aligned, uint32_t first_layer,
                                                      uint32_t last_layer) {
  if (!width || !height || !std::has_single_bit(bytes) || bytes > 16 || first_layer > last_layer)
    return std::nullopt;
  const uint32_t block_log2 = depth ? 17 : pipe_aligned ? 14 : 12;
  const uint32_t pixel_bits = depth ? 21 : block_log2 + 8 - std::countr_zero(bytes);
  const uint32_t xb = (pixel_bits + 1) / 2, yb = pixel_bits / 2;
  const uint64_t slice =
      ((uint64_t{width} + (1u << xb) - 1) >> xb) * ((uint64_t{height} + (1u << yb) - 1) >> yb)
      << block_log2;
  base &= ~((uint64_t{1} << block_log2) - 1);
  if (first_layer && slice > (UINT64_MAX - base) / first_layer)
    return std::nullopt;
  const uint64_t begin = base + first_layer * slice;
  const uint64_t layers = uint64_t{last_layer} - first_layer + 1;
  if (slice > (UINT64_MAX - begin) / layers)
    return std::nullopt;
  return VmRamRange{begin, layers * slice};
}

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
  static constexpr auto dcc_equations = [] {
    std::array<std::array<ImageAddressEquation, 5>, 2> result{};
    for (uint32_t mode = 0; mode < 2; ++mode)
      for (uint32_t element = 0; element < 5; ++element)
        result[mode][element] = make_image_equation(dcc_masks[mode][element], 14);
    return result;
  }();
  static constexpr auto unaligned_equations = [] {
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

/// Read the current D32 HTILE words in draw order without per-word VM admission.
/// The caller excludes observers and debugging. Only private RAM with a stable
/// strict-transport span is admitted; refusal has no stores or fault callbacks.
/// A non-expanded word must use the original loop, not a saved clear key: an
/// earlier pixel store can alias a later metadata word.
inline bool try_gfx11_expanded_htile(const GpuVmAccess &memory, uint64_t metadata, uint32_t width,
                                     uint32_t height, uint32_t swizzle) {
  struct RestoreErrno {
    int value = errno;
    ~RestoreErrno() { errno = value; }
  } restore_errno;
  // Match the bounded graphics attachment dimensions. This is not a general
  // image-transfer shortcut; shader materialization retains its own ordering.
  if (!width || !height || width > 4096 || height > 4096)
    return false;
  std::vector<uint64_t> addresses;
  addresses.reserve(size_t{(width + 7) / 8} * ((height + 7) / 8));
  uint64_t begin = UINT64_MAX, end = 0;
  for (uint32_t y = 0; y < height; y += 8) {
    for (uint32_t x = 0; x < width; x += 8) {
      const auto address = gfx11_metadata_address(metadata, x, y, width, height, 4, swizzle, true);
      if (!address || *address > UINT64_MAX - 4)
        return false;
      addresses.push_back(*address);
      begin = std::min(begin, *address);
      end = std::max(end, *address + 4);
    }
  }
  // All storage, including the lease's prepared request, precedes admission.
  // The envelope may include unused gaps, but only original logical words are
  // read. Unknown or fragmented backing declines before any bytes are copied.
  auto lease = memory.try_lease_ram(begin, end - begin);
  if (!lease)
    return false;
  const auto bytes = lease->bytes();
  for (uint64_t address : addresses) {
    uint32_t key;
    std::memcpy(&key, bytes.data() + (address - begin), sizeof(key));
    if ((key & 15) != 15)
      return false;
  }
  return true;
}

/// Inspect one color layer's current DCC keys without per-byte VM admission.
/// As with HTILE, the caller excludes observers and debugging. A non-expanded
/// key must restart the scalar loop: pixel stores can alias subsequent keys.
inline bool try_gfx11_expanded_dcc(const GpuVmAccess &memory, uint64_t metadata, uint32_t width,
                                   uint32_t height, uint32_t bytes, uint32_t swizzle,
                                   bool pipe_aligned, uint32_t layer) {
  struct RestoreErrno {
    int value = errno;
    ~RestoreErrno() { errno = value; }
  } restore_errno;
  if (!width || !height || width > 4096 || height > 4096 || !std::has_single_bit(bytes) ||
      bytes > 16)
    return false;
  const uint32_t bits = 8 - std::countr_zero(bytes);
  const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
  std::vector<uint64_t> addresses;
  addresses.reserve(size_t{(width + bw - 1) / bw} * ((height + bh - 1) / bh));
  uint64_t begin = UINT64_MAX, end = 0;
  for (uint32_t y = 0; y < height; y += bh) {
    for (uint32_t x = 0; x < width; x += bw) {
      const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle,
                                                  false, pipe_aligned, layer);
      if (!address || *address == UINT64_MAX)
        return false;
      addresses.push_back(*address);
      begin = std::min(begin, *address);
      end = std::max(end, *address + 1);
    }
  }
  // Acquire a fresh strict RAM lease after preparing all storage. Read only
  // the original key bytes; the enclosing range may contain unused gaps.
  auto lease = memory.try_lease_ram(begin, end - begin);
  if (!lease)
    return false;
  const auto values = lease->bytes();
  for (uint64_t address : addresses)
    if (values[address - begin] != std::byte{0xff})
      return false;
  return true;
}

namespace image_metadata_detail {

// Relative start of a 64/256 KiB GFX11 pixel swizzle block. Callers qualify the
// tiled swizzle and power-of-two element size first. DCC clear dimensions cover
// 256 bytes and HTILE clears cover 8x8 pixels; their aligned dimensions divide
// these larger block dimensions. The address equation and layer XOR only
// permute bytes inside the block, so every pixel of a selected clear stays here.
inline uint64_t pixel_swizzle_block_offset(uint32_t x, uint32_t y, uint32_t width, uint32_t bytes,
                                           uint32_t swizzle) {
  const uint32_t block_log2 = image_block_log2(false, swizzle);
  const uint32_t bits = block_log2 - std::countr_zero(bytes);
  const uint32_t xb = (bits + 1) / 2, yb = bits / 2;
  const uint64_t pitch = (uint64_t{width} + (1u << xb) - 1) >> xb;
  return (uint64_t{y >> yb} * pitch + (x >> xb)) << block_log2;
}

// One instruction's immutable descriptor owns this pure address cache. Every
// lookup still reads the current metadata key and executes the original core;
// only the integer equation is reused for coordinates in the same clear block.
class MetadataAddressCache {
public:
  MetadataAddressCache(uint64_t metadata, uint32_t width, uint32_t height, uint32_t bytes,
                       uint32_t swizzle, bool depth, bool pipe_aligned)
      : metadata_(metadata), width_(width), height_(height), bytes_(bytes), swizzle_(swizzle),
        depth_(depth), pipe_aligned_(pipe_aligned) {
    if (std::has_single_bit(bytes) && bytes <= 16) {
      const uint32_t bits = depth ? 6 : 8 - std::countr_zero(bytes);
      x_mask_ = (1u << ((bits + 1) / 2)) - 1;
      y_mask_ = (1u << (bits / 2)) - 1;
    }
  }

  std::optional<uint64_t> lookup(uint32_t x, uint32_t y, uint32_t layer = 0) {
    if (x >= width_ || y >= height_)
      return std::nullopt;
    const uint32_t block_x = x & ~x_mask_, block_y = y & ~y_mask_;
    if (valid_ && block_x == x_ && block_y == y_ && layer == layer_)
      return address_;
    const auto address = gfx11_metadata_address(metadata_, x, y, width_, height_, bytes_, swizzle_,
                                                depth_, pipe_aligned_, layer);
    if (address) {
      x_ = block_x;
      y_ = block_y;
      layer_ = layer;
      address_ = *address;
      valid_ = true;
    }
    return address;
  }

private:
  const uint64_t metadata_;
  const uint32_t width_, height_, bytes_, swizzle_;
  const bool depth_, pipe_aligned_;
  uint32_t x_mask_ = 0, y_mask_ = 0;
  uint32_t x_ = 0, y_ = 0, layer_ = 0;
  uint64_t address_ = 0;
  bool valid_ = false;
};

// Static error text defers exception allocation until an optional RAM lease
// ends.
/// Materialize one DCC clear block, retaining the uncompressed metadata
/// encoding. General delta compression is never produced by the functional
/// renderer.
template <typename Memory>
const char *
materialize_gfx11_dcc_at_address(const Memory &memory, uint64_t base, uint64_t address, uint32_t x,
                                 uint32_t y, uint32_t width, uint32_t height, uint32_t bytes,
                                 uint32_t swizzle, uint32_t layer = 0, uint64_t slice_size = 0,
                                 uint32_t pitch = 0, uint32_t tail_x = 0, uint32_t tail_y = 0) {
  if (!pitch)
    pitch = width;
  base = image_layer_base(false, base, slice_size, layer, bytes, swizzle);
  uint8_t key;
  if (memory.read(address, std::as_writable_bytes(std::span{&key, 1})) != VmAccessOutcome::Complete)
    return "image read failed";
  if (key == 0xff)
    return nullptr;
  const uint32_t bits = 8 - std::countr_zero(bytes);
  const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
  x &= ~(bw - 1);
  y &= ~(bh - 1);
  std::array<uint8_t, 16> value{};
  if (key == 1) {
    const auto clear = gfx11_image_address(base, x + tail_x, y + tail_y, pitch, bytes, swizzle);
    if (!clear)
      return "unsupported GFX11 DCC clear layout";
    if (memory.read(*clear, std::as_writable_bytes(std::span{value}.first(bytes))) !=
        VmAccessOutcome::Complete)
      return "image read failed";
  } else if (key == 2) {
    value.fill(0xff);
  } else if (key == 4 || key == 6) {
    const uint32_t step = key == 4 ? 2 : 4;
    const uint32_t one = key == 4 ? 0x3c00 : 0x3f800000;
    if (bytes % step)
      return "unsupported GFX11 DCC floating clear format";
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = one >> (8 * (i % step));
  } else if (key == 8 || key == 10) {
    if (bytes != 2 && bytes != 4 && bytes != 8)
      return "unsupported GFX11 DCC mixed clear format";
    const uint32_t last_component = bytes == 2 ? 1 : 3 * bytes / 4;
    for (uint32_t i = 0; i < bytes; ++i)
      value[i] = ((i >= last_component) == (key == 8)) ? 0xff : 0;
  } else if (key != 0) {
    return "unsupported GFX11 DCC compressed block";
  }
  for (uint32_t py = y; py < std::min(y + bh, height); ++py)
    for (uint32_t px = x; px < std::min(x + bw, width); ++px)
      if (memory.write(*gfx11_image_address(base, px + tail_x, py + tail_y, pitch, bytes, swizzle),
                       std::as_bytes(std::span{value}.first(bytes))) != VmAccessOutcome::Complete)
        return "image write failed";
  key = 0xff;
  if (memory.write(address, std::as_bytes(std::span{&key, 1})) != VmAccessOutcome::Complete)
    return "image write failed";
  return nullptr;
}

template <typename Memory>
const char *materialize_gfx11_dcc(const Memory &memory, uint64_t base, uint64_t metadata,
                                  uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                  uint32_t bytes, uint32_t swizzle, bool pipe_aligned = true,
                                  uint32_t layer = 0, uint64_t slice_size = 0) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, false,
                                              pipe_aligned, layer);
  if (!address)
    return "unsupported GFX11 DCC surface layout";
  return materialize_gfx11_dcc_at_address(memory, base, *address, x, y, width, height, bytes,
                                          swizzle, layer, slice_size);
}

/// Materialize a selected mip without touching adjacent levels or array slices.
template <typename Memory>
const char *materialize_gfx11_dcc_mip(const Memory &memory, uint64_t base, uint64_t metadata,
                                      uint32_t x, uint32_t y, uint32_t bytes, uint32_t swizzle,
                                      bool pipe_aligned, uint32_t layer,
                                      const Gfx11DccMipLayout &mip) {
  if (!mip.enabled)
    return nullptr;
  const uint32_t metadata_mask = (1u << (pipe_aligned ? 14 : 12)) - 1;
  metadata += mip.offset + layer * mip.slice_size;
  if (pipe_aligned)
    metadata ^= gfx11_image_slice_xor(layer, bytes, swizzle) & metadata_mask;
  const auto address = gfx11_metadata_address(metadata, x, y, mip.width, mip.height, bytes, swizzle,
                                              false, pipe_aligned);
  if (!address)
    return "unsupported GFX11 DCC mip layout";
  const auto &pixels = mip.pixels;
  return materialize_gfx11_dcc_at_address(
      memory, base + pixels.offset, *address, x, y, pixels.width, pixels.height, bytes, swizzle,
      layer, pixels.slice_size, pixels.pitch, pixels.tail_x, pixels.tail_y);
}

/// HTILE ZMask zero references DB_DEPTH_CLEAR; ZMask fifteen is uncompressed.
template <typename Memory>
const char *materialize_gfx11_htile_at_address(const Memory &memory, uint64_t base,
                                               uint64_t address, uint32_t x, uint32_t y,
                                               uint32_t width, uint32_t height, uint32_t bytes,
                                               uint32_t swizzle,
                                               std::optional<uint32_t> clear_bits = std::nullopt,
                                               bool has_stencil = false) {
  if (bytes != 1 && bytes != 2 && bytes != 4)
    return "unsupported GFX11 HTILE surface layout";
  uint32_t key;
  if (memory.read(address, std::as_writable_bytes(std::span{&key, 1})) != VmAccessOutcome::Complete)
    return "image read failed";
  if (bytes == 1) {
    if ((key & 0x300u) != 0x300u)
      return "GFX11 stencil HTILE clear requires a stencil clear register";
    return nullptr;
  }
  if ((key & 15) == 15)
    return nullptr;
  if (key & 15)
    return "unsupported GFX11 HTILE compressed block";
  if (!clear_bits) {
    // Texture-compatible fast clears encode the endpoints exactly. Other
    // clears require DB_DEPTH_CLEAR, which an image descriptor cannot supply.
    if (key == 0)
      clear_bits = 0;
    else if (key == 0xfffffff0)
      clear_bits = bytes == 2 ? 65535 : 0x3f800000;
    else
      return "GFX11 HTILE clear requires a depth clear register";
  }
  x &= ~7u;
  y &= ~7u;
  for (uint32_t py = y; py < std::min(y + 8, height); ++py)
    for (uint32_t px = x; px < std::min(x + 8, width); ++px)
      if (memory.write(*gfx11_image_address(base, px, py, width, bytes, swizzle),
                       std::as_bytes(std::span{&*clear_bits, 1}).first(bytes)) !=
          VmAccessOutcome::Complete)
        return "image write failed";
  key = has_stencil ? (key & 0x3f0u) | 0xfffff00fu : 0xfffc000fu;
  if (memory.write(address, std::as_bytes(std::span{&key, 1})) != VmAccessOutcome::Complete)
    return "image write failed";
  return nullptr;
}

template <typename Memory>
const char *materialize_gfx11_htile(const Memory &memory, uint64_t base, uint64_t metadata,
                                    uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                    uint32_t bytes, uint32_t swizzle,
                                    std::optional<uint32_t> clear_bits = std::nullopt,
                                    bool has_stencil = false) {
  const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle, true);
  if (!address)
    return "unsupported GFX11 HTILE surface layout";
  return materialize_gfx11_htile_at_address(memory, base, *address, x, y, width, height, bytes,
                                            swizzle, clear_bits, has_stencil);
}

} // namespace image_metadata_detail

inline void materialize_gfx11_dcc(const GpuVmAccess &memory, uint64_t base, uint64_t metadata,
                                  uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                  uint32_t bytes, uint32_t swizzle, bool pipe_aligned = true,
                                  uint32_t layer = 0, uint64_t slice_size = 0) {
  if (const char *error = image_metadata_detail::materialize_gfx11_dcc(
          memory, base, metadata, x, y, width, height, bytes, swizzle, pipe_aligned, layer,
          slice_size))
    throw std::runtime_error(error);
}

inline void materialize_gfx11_htile(const GpuVmAccess &memory, uint64_t base, uint64_t metadata,
                                    uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                    uint32_t bytes, uint32_t swizzle,
                                    std::optional<uint32_t> clear_bits = std::nullopt,
                                    bool has_stencil = false) {
  if (const char *error = image_metadata_detail::materialize_gfx11_htile(
          memory, base, metadata, x, y, width, height, bytes, swizzle, clear_bits, has_stencil))
    throw std::runtime_error(error);
}

namespace image_metadata_detail {

// This adapter executes the same scalar block/pixel/key order as ordinary VM
// accesses. The joint lease has already proved every access is private RAM.
class RamAccess {
public:
  RamAccess(const std::array<VmRamRange, 2> &ranges, const VmRamLease &lease)
      : ranges_(ranges), bytes_{lease.bytes(0), lease.bytes(1)} {}
  bool has_ram() const { return true; }
  VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes) const {
    const auto *source = pointer(address, bytes.size());
    if (!source)
      return VmAccessOutcome::Faulted;
    std::memcpy(bytes.data(), source, bytes.size());
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const {
    auto *destination = pointer(address, bytes.size());
    if (!destination)
      return VmAccessOutcome::Faulted;
    std::memcpy(destination, bytes.data(), bytes.size());
    return VmAccessOutcome::Complete;
  }

private:
  std::byte *pointer(uint64_t address, size_t size) const {
    for (size_t i = 0; i < ranges_.size(); ++i)
      if (address >= ranges_[i].address && address - ranges_[i].address <= bytes_[i].size() &&
          size <= bytes_[i].size() - (address - ranges_[i].address))
        return bytes_[i].data() + address - ranges_[i].address;
    return nullptr;
  }
  const std::array<VmRamRange, 2> &ranges_;
  std::array<std::span<std::byte>, 2> bytes_;
};

// Shader transfers initially pin only metadata and direct texels. A compressed
// clear may reach beyond those spans. Refuse that whole access without effects,
// invalidate every borrowed view, and release the retained guards before the
// original VM operation. Nothing is replayed, and later accesses stay ordinary.
class FallbackRamAccess {
public:
  FallbackRamAccess(const GpuVmAccess &memory, const std::array<VmRamRange, 2> &ranges,
                    VmRamLeaseRequest &request)
      : memory_(memory), request_(request) {
    try {
      ram_.emplace(ranges, request);
    } catch (...) {
      release();
      throw;
    }
  }
  FallbackRamAccess(const FallbackRamAccess &) = delete;
  FallbackRamAccess &operator=(const FallbackRamAccess &) = delete;
  ~FallbackRamAccess() {
    if (ram_)
      release();
  }
  bool has_ram() const { return ram_.has_value(); }
  VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes) const {
    if (ram_) {
      if (ram_->read(address, bytes) == VmAccessOutcome::Complete)
        return VmAccessOutcome::Complete;
      release();
    }
    return memory_.read(address, bytes);
  }
  VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const {
    if (ram_) {
      if (ram_->write(address, bytes) == VmAccessOutcome::Complete)
        return VmAccessOutcome::Complete;
      release();
    }
    return memory_.write(address, bytes);
  }

private:
  void release() const {
    ram_.reset();
    const int saved_errno = errno;
    request_.release();
    errno = saved_errno;
  }
  const GpuVmAccess &memory_;
  VmRamLeaseRequest &request_;
  mutable std::optional<RamAccess> ram_;
};

// No key is read until both complete envelopes are admitted. Unknown mappings,
// partial extents and physical aliases decline before effects. Lease admission
// pins mappings, not contents: keys and key-1 clear values are read at their
// original positions, and each expanded key follows all of its pixel stores.
template <bool Depth>
bool try_materialize_layer(const GpuVmAccess &memory, uint64_t base, uint64_t metadata,
                           uint32_t width, uint32_t height, uint32_t bytes, uint32_t swizzle,
                           bool pipe_aligned, uint32_t layer, uint64_t slice_size,
                           uint32_t clear_bits, bool has_stencil = false) {
  // A negative immutable capability avoids new prepare callbacks on default
  // custom transports; positive eligibility still requires the live joint
  // lease.
  if (!memory.supports_ram_word_reads() || !width || !height || width > 4096 || height > 4096 ||
      (Depth ? (bytes != 4 || (swizzle != 24 && swizzle != 28))
             : (!std::has_single_bit(bytes) || bytes > 16 || (swizzle != 27 && swizzle != 31))))
    return false;
  const auto mip = image_mip_layout(false, swizzle, bytes, width, height, 1, 0);
  if (!mip || (layer && (!slice_size || slice_size > (UINT64_MAX - base) / layer)))
    return false;
  const uint32_t pixel_block_log2 = image_block_log2(false, swizzle);
  const uint64_t pixel_base = image_layer_base(false, base, slice_size, layer, bytes, swizzle) &
                              ~((uint64_t{1} << pixel_block_log2) - 1);
  if (mip->slice_size > UINT64_MAX - pixel_base)
    return false;
  // The image equation permutes bytes within each padded block; layer XOR is
  // confined to that block too. The single-level padded slice bounds all
  // pixels.
  const auto metadata_range =
      gfx11_metadata_range(metadata, width, height, bytes, Depth, pipe_aligned, layer, layer);
  if (!metadata_range)
    return false;
  const uint64_t metadata_layer = metadata_range->address;
  const uint64_t metadata_slice = metadata_range->size;
  const uint32_t bits = 8 - std::countr_zero(bytes);
  const uint32_t bw = Depth ? 8 : 1u << ((bits + 1) / 2), bh = Depth ? 8 : 1u << (bits / 2);
  constexpr uint32_t key_bytes = Depth ? 4 : 1;
  uint64_t begin = UINT64_MAX, end = 0;
  for (uint32_t y = 0; y < height; y += bh) {
    for (uint32_t x = 0; x < width; x += bw) {
      const auto address = gfx11_metadata_address(metadata, x, y, width, height, bytes, swizzle,
                                                  Depth, pipe_aligned, layer);
      if (!address || *address < metadata_layer ||
          *address - metadata_layer > metadata_slice - key_bytes)
        return false;
      begin = std::min(begin, *address);
      end = std::max(end, *address + key_bytes);
    }
  }
  const std::array<VmRamRange, 2> ranges{{{begin, end - begin}, {pixel_base, mip->slice_size}}};
  const int saved_errno = errno;
  auto lease = memory.try_lease_ram(ranges);
  errno = saved_errno;
  if (!lease)
    return false;
  const RamAccess ram(ranges, *lease);
  const char *error = nullptr;
  for (uint32_t y = 0; y < height && !error; y += bh) {
    for (uint32_t x = 0; x < width && !error; x += bw) {
      if constexpr (Depth)
        error = materialize_gfx11_htile(ram, base, metadata, x, y, width, height, bytes, swizzle,
                                        clear_bits, has_stencil);
      else
        error = materialize_gfx11_dcc(ram, base, metadata, x, y, width, height, bytes, swizzle,
                                      pipe_aligned, layer, slice_size);
    }
  }
  lease.reset();
  errno = saved_errno;
  // Preserve the successful prefix, then allocate/throw only after all guards
  // have ended. Unsupported later keys must never restart the ordinary loop.
  if (error)
    throw std::runtime_error(error);
  return true;
}

} // namespace image_metadata_detail

/// Caller opt-in excludes observers/debugging. An all-expanded probe comes
/// first so unchanged layers do not acquire or validate pixel backing
/// unnecessarily.
inline bool try_materialize_gfx11_dcc_layer(const GpuVmAccess &memory, uint64_t base,
                                            uint64_t metadata, uint32_t width, uint32_t height,
                                            uint32_t bytes, uint32_t swizzle, bool pipe_aligned,
                                            uint32_t layer, uint64_t slice_size) {
  return image_metadata_detail::try_materialize_layer<false>(
      memory, base, metadata, width, height, bytes, swizzle, pipe_aligned, layer, slice_size, 0);
}

/// D32-only depth materialization, preserving shared stencil metadata bits.
/// Stencil pixels retain their original separate materialization path.
inline bool try_materialize_gfx11_htile_layer(const GpuVmAccess &memory, uint64_t base,
                                              uint64_t metadata, uint32_t width, uint32_t height,
                                              uint32_t swizzle, uint32_t clear_bits,
                                              bool has_stencil = false) {
  return image_metadata_detail::try_materialize_layer<true>(
      memory, base, metadata, width, height, 4, swizzle, true, 0, 0, clear_bits, has_stencil);
}

/// Expand a fast stencil clear while preserving the shared depth metadata.
inline void materialize_gfx11_stencil_htile(const GpuVmAccess &memory, uint64_t base,
                                            uint64_t metadata, uint32_t x, uint32_t y,
                                            uint32_t width, uint32_t height, uint32_t depth_bytes,
                                            uint32_t depth_swizzle, uint32_t stencil_swizzle,
                                            uint8_t clear) {
  const auto address =
      gfx11_metadata_address(metadata, x, y, width, height, depth_bytes, depth_swizzle, true);
  if (!address)
    throw std::runtime_error("unsupported GFX11 stencil HTILE surface layout");
  uint32_t key;
  read_image_bytes(memory, *address, {reinterpret_cast<uint8_t *>(&key), 4});
  const uint32_t smem = (key >> 8) & 3;
  if (smem == 3)
    return;
  if (smem)
    throw std::runtime_error("unsupported GFX11 stencil HTILE compressed block");
  x &= ~7u;
  y &= ~7u;
  for (uint32_t py = y; py < std::min(y + 8, height); ++py)
    for (uint32_t px = x; px < std::min(x + 8, width); ++px)
      write_image_bytes(memory, *gfx11_image_address(base, px, py, width, 1, stencil_swizzle),
                        {&clear, 1});
  key |= 0x3f0u;
  write_image_bytes(memory, *address, {reinterpret_cast<const uint8_t *>(&key), 4});
}

} // namespace rocjitsu::amdgpu
#endif
