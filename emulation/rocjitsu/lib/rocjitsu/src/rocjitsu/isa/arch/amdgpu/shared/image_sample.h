// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_footprint.h"
#include "util/data_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief LOD input mode of a qualified sampling instruction.
enum class ImageSampleMode { Implicit, Zero, Explicit, Derivatives };

/// @brief Round texture coordinates to the sampler's eight fractional bits.
inline double image_sample_fixed8(double value) { return std::floor(value * 256.0 + 0.5) / 256.0; }

/// @brief Quantize sampler fractional inputs to nearest-even without host rounding-mode dependence.
inline double image_sample_round_even8(double value) {
  const double scaled = value * 256.0;
  const double lower = std::floor(scaled);
  const double rest = scaled - lower;
  return (lower + (rest > 0.5 || (rest == 0.5 && std::fmod(lower, 2.0) != 0))) / 256.0;
}

/// @brief Wrap modes 0/1 repeat/mirror, 6 returns a transparent-black border, others clamp.
/// @details Callers must reject unsupported wrap modes 4, 5 and 7 first.
inline std::optional<uint32_t> image_sample_coordinate(double value, uint32_t size, uint32_t wrap) {
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
    if (wrap == 3 && value < 0)
      value = -1 - value;
    value = std::clamp(value, 0.0, double(size - 1));
  }
  return static_cast<uint32_t>(value);
}

/// @brief Capture the texel footprint before any memory request or destination write.
inline util::Result prepare_image_sample(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg, uint32_t sampler_reg,
                                         const ImageAddressRegisters &vaddr, ImageFields fields,
                                         ImageSampleMode mode) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  const auto decoded = prepare_image_resource(wf, state, resource_reg, fields);
  if (decoded.failed())
    return util::Result::failure();
  if (fields.gather)
    state.buffer_components = 4;
  const auto &resource = decoded.value();
  if (!resource.width || !scalar_selector_range_is_backed(wf, sampler_reg, 4))
    return util::Result::success();
  if (resource.type == kImageCube || resource.type > kImage2dArray || resource.array_pitch ||
      resource.base_array)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx9 = arch_is_cdna_4_or_lower(wf.cu().arch());
  const bool gfx10 =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2;
  const bool gfx12 = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t type = gfx9 ? resource.type : fields.dim + kImage1d;
  // HIP tex1D may view a width-by-one 2D allocation. DIM supplies X only;
  // the missing Y coordinate is zero, including filtering and wrapping.
  const bool one_dimensional_view = (type == kImage1d && resource.type == kImage2d) ||
                                    (type == kImage1dArray && resource.type == kImage2dArray);
  if (type != resource.type && !one_dimensional_view)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool has_y = type != kImage1d && type != kImage1dArray;
  const bool volume = type == kImage3d;
  const bool layered = type == kImage1dArray || type == kImage2dArray;
  const bool has_slice = volume || (layered && (!gfx9 || fields.da));
  const uint32_t coordinates = 1 + has_y + has_slice;
  const uint32_t spatial_dimensions = 1 + has_y + volume;
  const uint32_t filter_dimensions = spatial_dimensions + one_dimensional_view;
  if (gfx9 && fields.a16 && mode == ImageSampleMode::Derivatives)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const uint32_t prefix_words = mode == ImageSampleMode::Derivatives ? 2 * spatial_dimensions : 0;
  const uint32_t inputs = coordinates + (mode == ImageSampleMode::Explicit);
  const uint32_t registers = prefix_words + (fields.a16 ? (inputs + 1) / 2 : inputs);
  if (fields.gather && (!has_y || volume || std::popcount(fields.dmask & 15u) != 1))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  for (uint32_t i = 0; i < registers; ++i)
    if (vaddr[i] >= wf.num_vgprs())
      return fail(InstructionExecutionError::UnsupportedOperandValue);
  std::array<uint32_t, 4> sampler;
  for (uint32_t i = 0; i < sampler.size(); ++i)
    sampler[i] = read_scalar_selector(wf, sampler_reg + i);
  const bool normalized = !fields.unorm && !(sampler[0] & (1u << 15));
  const bool truncate_coordinates = sampler[0] & (1u << 27);
  std::array<uint32_t, 3> wrap{sampler[0] & 7, (sampler[0] >> 3) & 7, (sampler[0] >> 6) & 7};
  for (uint32_t i = 0; i < filter_dimensions; ++i)
    if (wrap[i] > 3 && wrap[i] != 6)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  // GFX11/12 captures: unnormalized repeat clamps; mirror reflects once, then clamps.
  // CLR normally sanitizes this combination before constructing the sampler.
  if (!normalized)
    for (auto &axis : wrap)
      if (axis <= 1)
        axis += 2;
  const uint32_t mag_filter = (sampler[2] >> 20) & 3;
  const uint32_t min_filter = (sampler[2] >> 22) & 3;
  const uint32_t mip_filter = (sampler[2] >> 26) & 3;
  const uint32_t perf_z = sampler[1] >> 28;
  const uint32_t perf_mip =
      gfx12 ? (sampler[2] >> 30) | ((sampler[3] & 3) << 2) : (sampler[1] >> 24) & 15;
  if (mag_filter > 1 || min_filter > 1 || mag_filter != min_filter || mip_filter > 2 ||
      (sampler[3] >> 30) != 0 ||
      (sampler[0] &
       ((3u << 29) | (7u << 9) | (1u << 20) | (1u << 19) | (gfx9 ? 0u : (1u << 31)))) ||
      (sampler[2] & ((3u << 24) | (63u << 14) | (gfx9 || gfx10 ? 1u << 28 : 0))) ||
      resource.min_lod || resource.corner_samples ||
      (resource.perf_mod && (perf_mip || (volume && perf_z))))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!fields.gather && (mag_filter || min_filter || mip_filter == 2) &&
      (state.decoded_buffer_format.number == BufferNumberFormat::Uint ||
       state.decoded_buffer_format.number == BufferNumberFormat::Sint))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  auto access = std::make_unique<ImageSampleAccess>();
  access->gather = fields.gather;
  const bool multi_tap = fields.gather || mag_filter || min_filter || mip_filter == 2;
  const uint32_t spatial_taps = multi_tap ? (1u << filter_dimensions) : 1;
  const uint32_t mip_taps = mip_filter == 2 && !fields.gather ? 2 : 1;
  access->taps.resize(spatial_taps * mip_taps);
  RegisterAccess regs(wf);
  const auto read = [&](uint32_t component, uint32_t lane) {
    const uint32_t bits = regs.read_vgpr(
        wf.vgpr_alloc().base + vaddr[prefix_words + (fields.a16 ? component / 2 : component)],
        lane);
    return double(fields.a16 ? util::f16_to_f32((bits >> (16 * (component % 2))) & 0xffffu)
                             : std::bit_cast<float>(bits));
  };
  const bool storage_has_y = has_y || one_dimensional_view;
  const ImageLayout layout{.width = resource.width,
                           .height = storage_has_y ? resource.height : 1u,
                           .depth = layered || volume ? resource.depth : 1u,
                           .pitch = resource.pitch,
                           .element_bytes = state.elem_size,
                           .swizzle_mode = resource.swizzle_mode,
                           .dimensions = volume          ? 3u
                                         : storage_has_y ? 2u
                                                         : 1u,
                           .mip_levels = resource.max_level + 1};
  std::array<std::optional<ImageAddressing>, 17> addresses;
  addresses[resource.base_level] =
      ImageAddressing::prepare(wf.cu().image_address_config(), layout, resource.base_level);
  if (!addresses[resource.base_level] ||
      !addresses[resource.base_level]->has_supported_base(resource.base))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(state.exec_mask & (uint64_t{1} << lane)))
      continue;
    double lod = mode == ImageSampleMode::Explicit ? read(coordinates, lane) : 0;
    if (mode == ImageSampleMode::Derivatives ||
        (mode == ImageSampleMode::Implicit &&
         (mip_filter && resource.base_level != resource.last_level))) {
      std::array<double, 3> dx{}, dy{};
      const std::array<uint32_t, 3> extents{std::max(1u, resource.width >> resource.base_level),
                                            std::max(1u, resource.height >> resource.base_level),
                                            std::max(1u, resource.depth >> resource.base_level)};
      if (mode == ImageSampleMode::Implicit) {
        const uint32_t quad = lane & ~3u;
        // Inactive helper-lane coordinates have not been hardware-qualified.
        if ((state.exec_mask & (uint64_t{15} << quad)) != (uint64_t{15} << quad))
          return fail(InstructionExecutionError::UnimplementedInstruction);
        for (uint32_t axis = 0; axis < spatial_dimensions; ++axis) {
          dx[axis] = read(axis, quad + 1) - read(axis, quad);
          dy[axis] = read(axis, quad + 2) - read(axis, quad);
        }
      } else {
        for (uint32_t axis = 0; axis < spatial_dimensions; ++axis) {
          const auto gradient = [&](uint32_t component) {
            return double(std::bit_cast<float>(
                regs.read_vgpr(wf.vgpr_alloc().base + vaddr[component], lane)));
          };
          dx[axis] = gradient(axis);
          dy[axis] = gradient(axis + spatial_dimensions);
        }
      }
      lod = image_gradient_lod(dx, dy, normalized ? extents : std::array<uint32_t, 3>{1, 1, 1},
                               spatial_dimensions);
    }
    if (std::isnan(lod))
      lod = 0;
    else if (std::isinf(lod))
      lod = std::copysign(double(std::numeric_limits<float>::max()), lod);
    const uint32_t lod_mask = gfx12 ? 0x1fffu : 0xfffu;
    const double minimum_lod = (sampler[1] & lod_mask) / 256.0;
    const double maximum_lod = ((sampler[1] >> (gfx12 ? 13 : 12)) & lod_mask) / 256.0;
    const int32_t bias = int32_t((sampler[2] & 0x3fffu) ^ 0x2000u) - 0x2000;
    lod = image_sample_round_even8(lod) + bias / 256.0;
    lod = std::clamp(lod, minimum_lod, std::max(minimum_lod, maximum_lod));
    lod = mip_filter ? std::clamp(lod, 0.0, double(resource.last_level - resource.base_level)) : 0;
    const uint32_t first_level =
        resource.base_level + uint32_t(std::floor(lod + (mip_filter == 1 ? 0.5 : 0)));
    const double mip_fraction = mip_filter == 2 ? image_sample_fixed8(lod - std::floor(lod)) : 0;
    if (gfx12 && mip_fraction != 0)
      access->mip_blend_mask |= uint64_t{15} << (lane & ~3u);
    const bool linear = fields.gather || (lod <= 0 ? mag_filter : min_filter);
    double layer = layered && has_slice ? read(has_y ? 2 : 1, lane) : 0;
    if (std::isnan(layer))
      layer = 0;
    const double lower_layer = std::floor(layer);
    const double layer_fraction = layer - lower_layer;
    const double rounded_layer =
        lower_layer +
        (layer_fraction > 0.5 || (layer_fraction == 0.5 && std::fmod(lower_layer, 2.0) != 0));
    const uint32_t array_layer =
        static_cast<uint32_t>(std::clamp(rounded_layer, 0.0, double(resource.depth - 1)));
    for (uint32_t mip_tap = 0; mip_tap < mip_taps; ++mip_tap) {
      const uint32_t level = std::min(resource.last_level, first_level + mip_tap);
      if (!addresses[level])
        addresses[level] = ImageAddressing::prepare(wf.cu().image_address_config(), layout, level);
      if (!addresses[level])
        return fail(InstructionExecutionError::UnimplementedInstruction);
      std::array<uint32_t, 3> sizes{std::max(1u, resource.width >> level),
                                    std::max(1u, resource.height >> level),
                                    std::max(1u, resource.depth >> level)};
      std::array<double, 3> origins{}, fractions{};
      for (uint32_t axis = 0; axis < filter_dimensions; ++axis) {
        double value = axis < spatial_dimensions ? read(axis, lane) : 0;
        if (std::isnan(value))
          value = 0;
        else if (std::isinf(value))
          value = std::copysign(double(std::numeric_limits<float>::max()), value);
        if (wrap[axis] <= 1 && std::abs(value) >= 0x1p32)
          value =
              std::fmod(value, double(normalized ? 1 : sizes[axis]) * (wrap[axis] == 1 ? 2 : 1));
        value *= normalized ? sizes[axis] : 1;
        const double position = value - (linear ? 0.5 : 0);
        origins[axis] =
            std::floor(!linear && !truncate_coordinates ? image_sample_fixed8(position) : position);
        fractions[axis] = linear ? image_sample_round_even8(position - origins[axis]) : 0;
        // Duplicate texels collapse before interpolation. image_sample_values
        // returns a lone weight-1 tap's raw payload to preserve signed zero;
        // GFX12 fractional-mip quads explicitly disable that shortcut.
        const auto lower = image_sample_coordinate(origins[axis], sizes[axis], wrap[axis]);
        if (lower && lower == image_sample_coordinate(origins[axis] + 1, sizes[axis], wrap[axis]))
          fractions[axis] = 0;
      }
      for (uint32_t tap = 0; tap < spatial_taps; ++tap) {
        auto &destination = access->taps[mip_tap * spatial_taps + tap];
        double weight = mip_filter == 2 ? (mip_tap ? mip_fraction : 1 - mip_fraction) : 1;
        std::array<uint32_t, 3> coordinate{};
        bool border = false;
        for (uint32_t axis = 0; axis < filter_dimensions; ++axis) {
          const bool upper = (tap >> axis) & 1;
          weight *= upper ? fractions[axis] : 1 - fractions[axis];
          const auto selected =
              image_sample_coordinate(origins[axis] + upper, sizes[axis], wrap[axis]);
          border |= !selected;
          coordinate[axis] = selected.value_or(0);
        }
        if (fields.gather)
          weight = 1;
        destination.weights[lane] = weight;
        if (border || weight == 0)
          continue;
        const uint32_t z = layered ? array_layer : volume ? coordinate[2] : 0;
        if (!addresses[level]->contains(coordinate[0], storage_has_y ? coordinate[1] : 0, z))
          return fail(InstructionExecutionError::UnsupportedOperandValue);
        const uint64_t offset =
            addresses[level]->byte_offset(coordinate[0], storage_has_y ? coordinate[1] : 0, z);
        destination.addresses[lane] = addr_calc::buffer_virtual_address(resource.base + offset);
        destination.lane_mask |= uint64_t{1} << lane;
        state.lane_mask |= uint64_t{1} << lane;
      }
    }
  }

  access->mip_blend_mask &= state.exec_mask;
  state.per_lane_addr = access->taps.front().addresses;
  state.image_sample = std::move(access);
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu
