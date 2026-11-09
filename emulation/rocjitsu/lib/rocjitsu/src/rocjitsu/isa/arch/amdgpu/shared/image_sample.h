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

enum class ImageClamp : uint32_t {
  Repeat = 0,
  Mirror = 1,
  Edge = 2,
  MirrorOnce = 3,
  Border = 6,
};

inline std::optional<uint32_t> image_sample_coordinate(double value, uint32_t size,
                                                       ImageClamp wrap) {
  if (wrap == ImageClamp::Repeat || wrap == ImageClamp::Mirror) {
    const double period = double(size) * (wrap == ImageClamp::Mirror ? 2 : 1);
    value = std::fmod(value, period);
    if (value < 0)
      value += period;
    if (value >= size)
      value = period - 1 - value;
  } else if (wrap == ImageClamp::Border) {
    if (value < 0 || value >= size)
      return std::nullopt;
  } else {
    if (wrap == ImageClamp::MirrorOnce && value < 0)
      value = -1 - value;
    value = std::clamp(value, 0.0, double(size - 1));
  }
  return static_cast<uint32_t>(value);
}

enum class ImageXyFilter : uint32_t { Point = 0, Linear = 1 };
enum class ImageMipFilter : uint32_t { None = 0, Point = 1, Linear = 2 };

struct ImageSampler {
  std::array<ImageClamp, 3> clamp;
  uint32_t min_lod_fixed8, max_lod_fixed8; ///< Unsigned 1/256-LOD units.
  int32_t lod_bias_fixed8;                 ///< Signed 1/256-LOD units.
  ImageXyFilter mag_filter, min_filter;
  ImageMipFilter mip_filter;
  uint32_t z_filter;
  uint32_t perf_mip, perf_z;
  uint32_t border_color_type, filter_mode, max_aniso_ratio, secondary_lod_bias;
  bool force_unnormalized, truncate_coordinates, force_degamma, mc_coord_trunc;
  bool skip_degamma, filter_precision_fix, derivative_adjust, mip_point_preclamp;
  bool compat_mode, blend_prt, aniso_override;
};

/// @brief Decode the sampler words used by the supported image families.
/// @details SQ_IMG_SAMP_WORD layouts follow ROCr resource_ai/gfx11/gfx12 and
/// PAL gfx10_sq_ko_reg.h. GFX12 reuses the high word-2 bits for PERF_MIP.
inline ImageSampler decode_image_sampler(const std::array<uint32_t, 4> &words,
                                         rj_code_arch_t arch) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t lod_bits = gfx12 ? 13 : 12;
  const uint32_t lod_mask = (1u << lod_bits) - 1;
  return {
      .clamp = {ImageClamp(words[0] & 7), ImageClamp((words[0] >> 3) & 7),
                ImageClamp((words[0] >> 6) & 7)},
      .min_lod_fixed8 = words[1] & lod_mask,
      .max_lod_fixed8 = (words[1] >> lod_bits) & lod_mask,
      .lod_bias_fixed8 = int32_t((words[2] & 0x3fffu) ^ 0x2000u) - 0x2000,
      .mag_filter = static_cast<ImageXyFilter>((words[2] >> 20) & 3),
      .min_filter = static_cast<ImageXyFilter>((words[2] >> 22) & 3),
      .mip_filter = static_cast<ImageMipFilter>((words[2] >> 26) & 3),
      .z_filter = (words[2] >> 24) & 3,
      .perf_mip = gfx12 ? (words[2] >> 30) | ((words[3] & 3) << 2) : (words[1] >> 24) & 15,
      .perf_z = words[1] >> 28,
      .border_color_type = words[3] >> 30,
      .filter_mode = (words[0] >> 29) & 3,
      .max_aniso_ratio = (words[0] >> 9) & 7,
      .secondary_lod_bias = (words[2] >> 14) & 63,
      .force_unnormalized = bool(words[0] & (1u << 15)),
      .truncate_coordinates = bool(words[0] & (1u << 27)),
      .force_degamma = bool(words[0] & (1u << 20)),
      .mc_coord_trunc = bool(words[0] & (1u << 19)),
      .skip_degamma = gfx9 ? bool(words[3] & (1u << 12)) : bool(words[0] & (1u << 31)),
      .filter_precision_fix = gfx9 && (words[2] & (1u << 30)),
      .derivative_adjust = !gfx9 && !gfx12 && (words[2] & (1u << 31)),
      .mip_point_preclamp = (gfx9 || gfx10) && (words[2] & (1u << 28)),
      .compat_mode = gfx9 && (words[0] & (1u << 31)),
      .blend_prt = !gfx12 && (words[2] & (1u << (gfx9 ? 29 : 30))),
      .aniso_override = bool(words[2] & (1u << (gfx9 ? 31 : 29))),
  };
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
  const util::FailureOr<ImageResource> decoded =
      prepare_image_resource(wf, state, resource_reg, fields);
  if (decoded.failed())
    return util::Result::failure();
  if (fields.gather)
    state.buffer_components = 4;
  const ImageResource &resource = decoded.value();
  if (!resource.width)
    return util::Result::success();
  if (resource.min_lod || resource.corner_samples)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx9 = arch_is_cdna_4_or_lower(wf.cu().arch());
  const bool gfx12 = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t type = gfx9 ? resource.type : fields.dim + kImage1d;
  // HIP tex1D may view a width-by-one 2D allocation. DIM supplies X only;
  // the missing Y coordinate is zero, including filtering and wrapping.
  const bool one_dimensional_view = (type == kImage1d && resource.type == kImage2d) ||
                                    (type == kImage1dArray && resource.type == kImage2dArray);
  if (type != resource.type && !one_dimensional_view)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const ImageCoordinateShape shape = image_coordinate_shape(type, !gfx9 || fields.da);
  const bool has_y = shape.has_y;
  const bool volume = shape.volume;
  const bool layered = shape.layered;
  const bool has_slice = shape.has_slice;
  const uint32_t coordinates = shape.coordinates();
  const uint32_t spatial_dimensions = shape.spatial_dimensions();
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
  ImageMipAddressCache addresses(wf.cu().image_address_config(), layout, resource.base);
  if (!addresses.get(resource.base_level))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!scalar_selector_range_is_backed(wf, sampler_reg, 4))
    return util::Result::success();
  std::array<uint32_t, 4> sampler_words;
  for (uint32_t word = 0; word < sampler_words.size(); ++word)
    sampler_words[word] = read_scalar_selector(wf, sampler_reg + word);
  const ImageSampler sampler = decode_image_sampler(sampler_words, wf.cu().arch());
  const bool normalized = !fields.unorm && !sampler.force_unnormalized;
  const bool truncate_coordinates = sampler.truncate_coordinates;
  std::array<ImageClamp, 3> wrap = sampler.clamp;
  for (uint32_t axis = 0; axis < filter_dimensions; ++axis)
    if (wrap[axis] > ImageClamp::MirrorOnce && wrap[axis] != ImageClamp::Border)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  // GFX11/12 captures: unnormalized repeat clamps; mirror reflects once, then clamps.
  // CLR normally sanitizes this combination before constructing the sampler.
  if (!normalized)
    for (ImageClamp &axis : wrap) {
      if (axis == ImageClamp::Repeat)
        axis = ImageClamp::Edge;
      else if (axis == ImageClamp::Mirror)
        axis = ImageClamp::MirrorOnce;
    }
  const ImageXyFilter mag_filter = sampler.mag_filter;
  const ImageXyFilter min_filter = sampler.min_filter;
  const ImageMipFilter mip_filter = sampler.mip_filter;
  if (mag_filter > ImageXyFilter::Linear || min_filter > ImageXyFilter::Linear ||
      mag_filter != min_filter || mip_filter > ImageMipFilter::Linear ||
      sampler.border_color_type || sampler.filter_mode || sampler.max_aniso_ratio ||
      sampler.force_degamma || sampler.mc_coord_trunc || sampler.skip_degamma || sampler.z_filter ||
      sampler.secondary_lod_bias || sampler.mip_point_preclamp || sampler.filter_precision_fix ||
      sampler.derivative_adjust || sampler.compat_mode || sampler.blend_prt ||
      sampler.aniso_override ||
      (resource.perf_mod && (sampler.perf_mip || (volume && sampler.perf_z))))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!fields.gather &&
      (mag_filter == ImageXyFilter::Linear || mip_filter == ImageMipFilter::Linear) &&
      (state.decoded_buffer_format.number == BufferNumberFormat::Uint ||
       state.decoded_buffer_format.number == BufferNumberFormat::Sint))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  auto access = std::make_unique<ImageSampleAccess>();
  access->gather = fields.gather;
  const bool multi_tap =
      fields.gather || mag_filter == ImageXyFilter::Linear || mip_filter == ImageMipFilter::Linear;
  const uint32_t spatial_taps = multi_tap ? (1u << filter_dimensions) : 1;
  const uint32_t mip_taps = mip_filter == ImageMipFilter::Linear && !fields.gather ? 2 : 1;
  access->taps.resize(spatial_taps * mip_taps);
  RegisterAccess regs(wf);
  const auto read = [&](uint32_t component, uint32_t lane) {
    const uint32_t bits = regs.read_vgpr(
        wf.vgpr_alloc().base + vaddr[prefix_words + (fields.a16 ? component / 2 : component)],
        lane);
    return double(fields.a16 ? util::f16_to_f32((bits >> (16 * (component % 2))) & 0xffffu)
                             : std::bit_cast<float>(bits));
  };
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(state.exec_mask & (uint64_t{1} << lane)))
      continue;
    double lod = mode == ImageSampleMode::Explicit ? read(coordinates, lane) : 0;
    if (mode == ImageSampleMode::Derivatives ||
        (mode == ImageSampleMode::Implicit &&
         (mip_filter != ImageMipFilter::None && resource.base_level != resource.last_level))) {
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
    const double minimum_lod = double(sampler.min_lod_fixed8) / kImageLodFractionScale;
    const double maximum_lod = double(sampler.max_lod_fixed8) / kImageLodFractionScale;
    const int32_t bias = sampler.lod_bias_fixed8;
    lod = image_sample_round_even8(lod) + double(bias) / kImageLodFractionScale;
    lod = std::clamp(lod, minimum_lod, std::max(minimum_lod, maximum_lod));
    lod = mip_filter != ImageMipFilter::None
              ? std::clamp(lod, 0.0, double(resource.last_level - resource.base_level))
              : 0;
    const uint32_t first_level =
        resource.base_level +
        uint32_t(std::floor(lod + (mip_filter == ImageMipFilter::Point ? 0.5 : 0)));
    const double mip_fraction =
        mip_filter == ImageMipFilter::Linear ? image_sample_fixed8(lod - std::floor(lod)) : 0;
    if (gfx12 && mip_fraction != 0)
      access->mip_blend_mask |= uint64_t{15} << (lane & ~3u);
    const bool linear = fields.gather || mag_filter == ImageXyFilter::Linear;
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
      const ImageAddressing *addressing = addresses.get(level);
      if (!addressing)
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
        if (wrap[axis] <= ImageClamp::Mirror && std::abs(value) >= 0x1p32)
          value = std::fmod(value, double(normalized ? 1 : sizes[axis]) *
                                       (wrap[axis] == ImageClamp::Mirror ? 2 : 1));
        value *= normalized ? sizes[axis] : 1;
        const double position = value - (linear ? 0.5 : 0);
        origins[axis] =
            std::floor(!linear && !truncate_coordinates ? image_sample_fixed8(position) : position);
        fractions[axis] = linear ? image_sample_round_even8(position - origins[axis]) : 0;
        // Duplicate texels collapse before interpolation. image_sample_values
        // returns a lone weight-1 tap's raw payload to preserve signed zero;
        // GFX12 fractional-mip quads explicitly disable that shortcut.
        const std::optional<uint32_t> lower =
            image_sample_coordinate(origins[axis], sizes[axis], wrap[axis]);
        if (lower && lower == image_sample_coordinate(origins[axis] + 1, sizes[axis], wrap[axis]))
          fractions[axis] = 0;
      }
      for (uint32_t tap = 0; tap < spatial_taps; ++tap) {
        auto &destination = access->taps[mip_tap * spatial_taps + tap];
        double weight =
            mip_filter == ImageMipFilter::Linear ? (mip_tap ? mip_fraction : 1 - mip_fraction) : 1;
        std::array<uint32_t, 3> coordinate{};
        bool border = false;
        for (uint32_t axis = 0; axis < filter_dimensions; ++axis) {
          const bool upper = (tap >> axis) & 1;
          weight *= upper ? fractions[axis] : 1 - fractions[axis];
          const std::optional<uint32_t> selected =
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
        if (!addressing->contains(coordinate[0], storage_has_y ? coordinate[1] : 0, z))
          return fail(InstructionExecutionError::UnsupportedOperandValue);
        const uint64_t offset =
            addressing->byte_offset(coordinate[0], storage_has_y ? coordinate[1] : 0, z);
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
