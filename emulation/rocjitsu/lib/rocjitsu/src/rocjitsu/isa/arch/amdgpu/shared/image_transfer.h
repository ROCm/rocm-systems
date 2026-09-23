// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_TRANSFER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_TRANSFER_H_

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {

enum class ImageSampleMode { Implicit, Zero, Explicit, Bias, Derivatives };

inline double round_sample_fixed8(double value) {
  const double scaled = value * 256, lo = std::floor(scaled);
  const double part = scaled - lo;
  return (lo + (part > 0.5 || (part == 0.5 && std::fmod(lo, 2.0) != 0))) / 256;
}

/// Prepare GFX11/12 image transfers and sampling, including A16 and 2D arrays.
inline bool prepare_image_transfer(Wavefront &wf, VectorMemState &d, uint32_t resource,
                                   uint32_t data, std::array<uint32_t, 7> coords, uint32_t dim,
                                   uint32_t mask, bool d16, bool unsupported_flags,
                                   uint32_t sampler = ~0u,
                                   ImageSampleMode sample_mode = ImageSampleMode::Implicit,
                                   bool a16 = false) {
  const auto unsupported = [&] {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  };
  const auto arch = wf.cu().arch();
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  if ((!gfx12 && arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
      unsupported_flags || (dim != 0 && dim != 1 && dim != 5) || !mask)
    return unsupported();
  std::array<uint32_t, 8> r{};
  for (uint32_t i = 0; i < r.size(); ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return false;
  for (uint32_t i = 0; i < r.size(); ++i)
    r[i] = read_scalar_selector(wf, resource + i);
  const uint32_t type = r[3] >> 28;
  const uint32_t swizzle = (r[3] >> 20) & 31;
  uint32_t width = ((r[1] >> 30) | ((r[2] & (gfx12 ? 0x3fff : 0xfff)) << 2)) + 1;
  uint32_t height = ((r[2] >> 14) & (gfx12 ? 0xffff : 0x3fff)) + 1;
  const uint32_t image_format = (r[1] >> (gfx12 ? 17 : 20)) & 255;
  const uint32_t format = image_format == 66 ? 42 : image_format;
  d.image_srgb = image_format == 66;
  const bool compressed = !gfx12 && (r[6] & (1u << 21));
  const uint32_t bytes = buffer_format_bytes(format);
  const uint32_t max_level = gfx12 ? (r[1] >> 12) & 31 : (r[1] >> 16) & 15;
  const uint32_t first_level = gfx12 ? (r[1] >> 25) & 31 : (r[3] >> 12) & 15;
  const uint32_t last_level = gfx12 ? (r[3] >> 15) & 31 : (r[3] >> 16) & 15;
  const bool sample = sampler != ~0u;
  d.image_sampling = sample;
  const uint32_t first_layer = (r[4] >> 16) & (gfx12 ? 0x3fff : 0x1fff);
  const uint32_t last_layer = r[4] & (gfx12 ? 0x3fff : 0x1fff);
  if ((type != 8 && type != 9 && type != 13) || (dim == 0 && type != 8) ||
      (type == 8 && (height != 1 || swizzle)) || (type != 13 && (r[4] >> 16)) || !bytes ||
      (type == 13 && (first_layer > last_layer || (r[4] & (gfx12 ? 0xc000c000u : 0xe000e000u)))) ||
      first_level > last_level || last_level > max_level ||
      (max_level && ((type != 13 && r[4]) || compressed)) ||
      (!d.is_load &&
       (d.image_srgb || (mask != 15 && !(mask == 1 && format >= 20 && format <= 22)))))
    return unsupported();
  const uint32_t resource_width = width, resource_height = height;
  const auto mip =
      image_mip_layout(gfx12, swizzle, bytes, width, height, max_level + 1, first_level);
  if (!mip)
    return unsupported();
  width = mip->width;
  height = mip->height;
  bool normalized = true;
  uint32_t min_filter = 0, mag_filter = 0, mip_filter = 0;
  double min_lod = 0, max_lod = 0, lod_bias = 0;
  uint32_t coordinate_offset = 0;
  uint32_t wrap_x = 2, wrap_y = 2;
  if (sample) {
    if ((dim != 1 && (dim != 5 || type != 13)) || !d.is_load)
      return unsupported();
    std::array<uint32_t, 4> s{};
    for (uint32_t i = 0; i < s.size(); ++i) {
      if (!scalar_selector_range_is_backed(wf, sampler + i, 1))
        return false;
      s[i] = read_scalar_selector(wf, sampler + i);
    }
    wrap_x = s[0] & 7;
    wrap_y = (s[0] >> 3) & 7;
    mag_filter = (s[2] >> 20) & 3;
    min_filter = (s[2] >> 22) & 3;
    mip_filter = (s[2] >> 26) & 3;
    const auto supported_wrap = [](uint32_t wrap) { return wrap <= 3 || wrap == 6; };
    if (!supported_wrap(wrap_x) || !supported_wrap(wrap_y) ||
        (s[0] & ((7u << 9) | (7u << 12) | (3u << 29) | (3u << 19))) || mag_filter > 1 ||
        min_filter > 1 || (s[2] & (3u << 24)) || mip_filter > 2)
      return unsupported();
    normalized = !(s[0] & (1u << 15));
    d.image_srgb &= !(s[0] & (1u << 31));
    min_lod = (s[1] & (gfx12 ? 0x1fff : 0xfff)) / 256.0;
    max_lod = ((s[1] >> (gfx12 ? 13 : 12)) & (gfx12 ? 0x1fff : 0xfff)) / 256.0;
    // The two signed bias fields must be sign-extended separately.
    lod_bias = (int32_t((s[2] & 0x3fff) ^ 0x2000) - 0x2000) / 256.0 +
               (int32_t(((s[2] >> 14) & 0x3f) ^ 0x20) - 0x20) / 16.0;
    // RGBA8 UNORM/sRGB and one-, two- or four-component FP16 filtering.
    const bool filterable = format == 42 || format == 13 || format == 29 || format == 57;
    if (min_lod > max_lod || ((min_filter || mag_filter || mip_filter == 2) && !filterable))
      return unsupported();
    coordinate_offset = sample_mode == ImageSampleMode::Bias          ? 1
                        : sample_mode == ImageSampleMode::Derivatives ? 4
                                                                      : 0;
    if (min_filter || mag_filter || mip_filter == 2 || wrap_x == 6 || wrap_y == 6) {
      if ((s[3] >> 30) == 3)
        return unsupported(); // Custom border-color tables.
      d.image_sample = std::make_unique<ImageSampleAccess>();
      d.image_sample->tap_count = mip_filter == 2 ? 8 : (min_filter || mag_filter ? 4 : 1);
      d.image_sample->border_color = s[3] >> 30;
    }
  }
  // The functional model retains uncompressed backing for image operations.
  // The memory pipeline materializes GFX11 metadata clears before each access.
  d.buffer_components = std::popcount(mask);
  d.buffer_d16 = d16;
  d.buffer_format = format;
  d.buffer_format_encoding = BufferFormatEncoding::Gfx11;
  uint32_t component = 0;
  for (uint32_t i = 0; i < 4; ++i)
    if (mask & (1u << i))
      d.buffer_selectors |= ((r[3] >> (3 * i)) & 7) << (3 * component++);
  d.wf_size = wf.wf_size();
  d.exec_mask = wf.exec();
  d.elem_size = bytes;
  d.num_elems = 1;
  d.dst_reg_base = wf.vgpr_alloc().base + data;
  const uint32_t registers = d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  if (data >= wf.num_vgprs() || registers > wf.num_vgprs() - data)
    return unsupported();
  const uint32_t body_components = 2 + (dim == 5) + (sample_mode == ImageSampleMode::Explicit);
  const uint32_t load_components = dim == 5 ? 3 : dim == 0 ? 1 : 2;
  const uint32_t coordinate_count =
      !sample ? (a16 ? (load_components + 1) / 2 : load_components)
              : coordinate_offset + (a16 ? (body_components + 1) / 2 : body_components);
  for (uint32_t i = 0; i < coordinate_count; ++i)
    if (coords[i] >= wf.num_vgprs())
      return unsupported();
  const uint64_t base =
      addr_calc::buffer_virtual_address(((uint64_t{r[1] & 255} << 32) | r[0]) << 8) + mip->offset;
  const uint32_t pitch_field = r[4] & (gfx12 ? 0xffff : 0x3fff);
  // Word 4 describes the last accessible layer for arrays, not custom pitch.
  const uint32_t pitch = type == 9 && swizzle == 0 && pitch_field ? pitch_field + 1 : mip->pitch;
  if (compressed) {
    const bool depth = swizzle == 24 || swizzle == 28;
    if ((type != 9 && type != 13) || (depth && type == 13) || max_level ||
        (depth ? (bytes != 2 && bytes != 4) : (swizzle != 27 && swizzle != 31)))
      return unsupported();
    d.image_metadata = std::make_unique<ImageMetadataAccess>();
    auto &image = *d.image_metadata;
    image.base = base;
    image.slice_size = mip->slice_size;
    image.metadata =
        addr_calc::buffer_virtual_address((uint64_t{r[7]} << 16) | (uint64_t{r[6] >> 24} << 8));
    image.width = width;
    image.height = height;
    image.swizzle = swizzle;
    image.pipe_aligned = r[6] & (1u << 19);
    image.depth = depth;
    if (depth && !image.pipe_aligned)
      return unsupported();
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    const auto load_coordinate = [&](uint32_t component) {
      const uint32_t value = wf.debug_read_vgpr(coords[a16 ? component / 2 : component], lane);
      return a16 ? (value >> (16 * (component % 2))) & 0xffff : value;
    };
    uint32_t x = sample ? 0 : load_coordinate(0);
    uint32_t y = sample || dim == 0 ? 0 : load_coordinate(1);
    if (sample) {
      const auto read = [&](uint32_t index, uint32_t source_lane) {
        if (a16 && index >= coordinate_offset) {
          const uint32_t component = index - coordinate_offset;
          const uint32_t packed =
              wf.debug_read_vgpr(coords[coordinate_offset + component / 2], source_lane);
          return util::f16_to_f32(static_cast<uint16_t>(packed >> (16 * (component % 2))));
        }
        return std::bit_cast<float>(wf.debug_read_vgpr(coords[index], source_lane));
      };
      double u = read(coordinate_offset, lane), v = read(coordinate_offset + 1, lane);
      if (!std::isfinite(u) || !std::isfinite(v))
        return unsupported();
      uint32_t layer = first_layer;
      if (dim == 5) {
        const double slice = read(coordinate_offset + 2, lane);
        if (!std::isfinite(slice))
          return unsupported();
        // RADV emits V_RNDNE_F32 for the floating-point slice before sampling.
        // Convert that integral value and clamp it to the resource view.
        layer += static_cast<uint32_t>(
            std::clamp(std::trunc(slice), 0.0, double(last_layer - first_layer)));
      }
      double lod = 0;
      if (sample_mode == ImageSampleMode::Explicit) {
        lod = read(coordinate_offset + 2 + (dim == 5), lane);
      } else if (sample_mode != ImageSampleMode::Zero && (max_level || min_filter != mag_filter)) {
        double dxu, dxv, dyu, dyv;
        if (sample_mode == ImageSampleMode::Derivatives) {
          dxu = read(0, lane);
          dxv = read(1, lane);
          dyu = read(2, lane);
          dyv = read(3, lane);
        } else {
          const uint32_t left = lane & ~1u, top = lane & ~2u;
          dxu = read(coordinate_offset, left + 1) - read(coordinate_offset, left);
          dxv = read(coordinate_offset + 1, left + 1) - read(coordinate_offset + 1, left);
          dyu = read(coordinate_offset, top + 2) - read(coordinate_offset, top);
          dyv = read(coordinate_offset + 1, top + 2) - read(coordinate_offset + 1, top);
        }
        if (!std::isfinite(dxu) || !std::isfinite(dxv) || !std::isfinite(dyu) ||
            !std::isfinite(dyv))
          return unsupported();
        const double sx = normalized ? width : 1, sy = normalized ? height : 1;
        const double rho = std::max(std::hypot(dxu * sx, dxv * sy), std::hypot(dyu * sx, dyv * sy));
        lod = rho > 0 ? std::log2(rho) : -32;
        lod += lod_bias;
        if (sample_mode == ImageSampleMode::Bias)
          lod += read(0, lane);
      }
      if (!std::isfinite(lod))
        return unsupported();
      lod = round_sample_fixed8(std::clamp(lod, min_lod, max_lod));
      const bool linear = (lod <= 0 ? mag_filter : min_filter) == 1;
      lod = mip_filter ? std::clamp(lod, 0.0, double(last_level - first_level)) : 0;
      const uint32_t level = first_level + uint32_t(std::floor(lod + (mip_filter == 1 ? 0.5 : 0)));
      if (wrap_x == 3)
        u = std::abs(u);
      if (wrap_y == 3)
        v = std::abs(v);
      const auto address_coordinate = [](double value, uint32_t size,
                                         uint32_t wrap) -> std::optional<uint32_t> {
        if (wrap <= 1) {
          const double period = double(size) * (wrap == 1 ? 2 : 1);
          value = std::fmod(value, period);
          if (value < 0)
            value += period;
          if (value >= size)
            value = period - 1 - value;
        } else if (wrap == 6) {
          if (value < 0 || value >= size)
            return std::nullopt;
        } else {
          value = std::clamp(value, 0.0, double(size - 1));
        }
        return static_cast<uint32_t>(value);
      };
      const auto lane_mip = image_mip_layout(gfx12, swizzle, bytes, resource_width, resource_height,
                                             max_level + 1, level);
      if (!lane_mip)
        return unsupported();
      const uint64_t resource_base = base - mip->offset;
      if (d.image_metadata)
        d.image_metadata->layers[lane] = layer;
      if (d.image_sample) {
        auto &access = *d.image_sample;
        const auto fraction = [](double value) {
          return static_cast<float>(round_sample_fixed8(value));
        };
        access.mip_fractions[lane] = fraction(lod - std::floor(lod));
        for (uint32_t tap = 0; tap < access.tap_count; ++tap) {
          const uint32_t mip_index = tap / 4;
          const auto selected =
              image_mip_layout(gfx12, swizzle, bytes, resource_width, resource_height,
                               max_level + 1, std::min(level + mip_index, last_level));
          if (!selected)
            return unsupported();
          double px = u * (normalized ? selected->width : 1) - (linear ? 0.5 : 0);
          double py = v * (normalized ? selected->height : 1) - (linear ? 0.5 : 0);
          // Clamp to texel centers before generating weights. This preserves
          // exact edge texels, including signed zero in a 1x1 mip level.
          if (linear && (wrap_x == 2 || wrap_x == 3))
            px = std::clamp(px, 0.0, double(selected->width - 1));
          if (linear && (wrap_y == 2 || wrap_y == 3))
            py = std::clamp(py, 0.0, double(selected->height - 1));
          const double x0 = std::floor(px), y0 = std::floor(py);
          access.fractions[lane][mip_index] =
              linear ? std::array{fraction(px - x0), fraction(py - y0)} : std::array{0.0f, 0.0f};
          const auto tx = address_coordinate(x0 + (linear ? tap & 1 : 0), selected->width, wrap_x);
          const auto ty =
              address_coordinate(y0 + (linear ? (tap >> 1) & 1 : 0), selected->height, wrap_y);
          if (!tx || !ty)
            continue;
          const uint64_t selected_base = image_layer_base(
              gfx12, resource_base + selected->offset, selected->slice_size, layer, bytes, swizzle);
          const auto address =
              gfx12 ? gfx12_image_address(selected_base, *tx + selected->tail_x,
                                          *ty + selected->tail_y,
                                          max_level ? selected->pitch : pitch, bytes, swizzle)
                    : gfx11_image_address(selected_base, *tx + selected->tail_x,
                                          *ty + selected->tail_y,
                                          max_level ? selected->pitch : pitch, bytes, swizzle);
          if (!address)
            return unsupported();
          access.taps[tap].addresses[lane] = *address;
          access.taps[tap].coordinates[lane] = *tx | (*ty << 16);
          access.taps[tap].lane_mask |= uint64_t{1} << lane;
        }
        d.per_lane_addr[lane] = access.taps[0].addresses[lane];
        d.lane_mask |= uint64_t{1} << lane;
        continue;
      }
      x = *address_coordinate(std::floor(u * (normalized ? lane_mip->width : 1)), lane_mip->width,
                              wrap_x);
      y = *address_coordinate(std::floor(v * (normalized ? lane_mip->height : 1)), lane_mip->height,
                              wrap_y);
      const uint64_t selected_base = image_layer_base(gfx12, resource_base + lane_mip->offset,
                                                      lane_mip->slice_size, layer, bytes, swizzle);
      const auto address =
          gfx12 ? gfx12_image_address(selected_base, x + lane_mip->tail_x, y + lane_mip->tail_y,
                                      max_level ? lane_mip->pitch : pitch, bytes, swizzle)
                : gfx11_image_address(selected_base, x + lane_mip->tail_x, y + lane_mip->tail_y,
                                      max_level ? lane_mip->pitch : pitch, bytes, swizzle);
      if (!address)
        return unsupported();
      d.per_lane_addr[lane] = *address;
      if (d.image_metadata)
        d.image_metadata->coordinates[lane] = x | (y << 16);
      d.lane_mask |= uint64_t{1} << lane;
      continue;
    }
    const uint32_t relative_layer = dim == 5 ? load_coordinate(2) : 0;
    if (type != 13 && relative_layer)
      return unsupported();
    if (x >= width || y >= height || (type == 13 && relative_layer > last_layer - first_layer))
      continue;
    const uint32_t layer = type == 13 ? first_layer + relative_layer : 0;
    const uint64_t layer_base =
        image_layer_base(gfx12, base, mip->slice_size, layer, bytes, swizzle);
    const auto address = gfx12 ? gfx12_image_address(layer_base, x + mip->tail_x, y + mip->tail_y,
                                                     pitch, bytes, swizzle)
                               : gfx11_image_address(layer_base, x + mip->tail_x, y + mip->tail_y,
                                                     pitch, bytes, swizzle);
    if (!address)
      return unsupported();
    d.per_lane_addr[lane] = *address;
    if (d.image_metadata) {
      d.image_metadata->coordinates[lane] = x | (y << 16);
      d.image_metadata->layers[lane] = layer;
    }
    d.lane_mask |= uint64_t{1} << lane;
  }
  if (!d.is_load)
    capture_buffer_format_store(wf, d, d.dst_reg_base);
  return true;
}

} // namespace rocjitsu::amdgpu

#endif
