// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_footprint.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_sampler.h"
#include "rocjitsu/vm/graphics/image_cube.h"
#include "rocjitsu/vm/graphics/image_filter.h"
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

/// @brief Round texture coordinates to the sampler's eight fractional bits.
inline double image_sample_fixed8(double value) { return std::floor(value * 256.0 + 0.5) / 256.0; }

/// @brief Quantize sampler fractional inputs to nearest-even without host rounding-mode dependence.
inline double image_sample_round_even8(double value) {
  const double scaled = value * 256.0;
  const double lower = std::floor(scaled);
  const double rest = scaled - lower;
  return (lower + (rest > 0.5 || (rest == 0.5 && std::fmod(lower, 2.0) != 0))) / 256.0;
}

/// GFX11/12 apply the shader CL operand as a per-lane minimum in 8-bit fixed point.
inline std::optional<double> image_shader_min_lod(double lod, double minimum) {
  if (!std::isfinite(minimum))
    return std::nullopt;
  return std::max(lod, image_sample_round_even8(minimum));
}

inline std::optional<uint32_t> image_sample_coordinate(double value, uint32_t size,
                                                       ImageClamp wrap) {
  if (wrap == ImageClamp::Repeat || wrap == ImageClamp::Mirror) {
    const double period = double(size) * (wrap == ImageClamp::Mirror ? 2 : 1);
    value = std::fmod(value, period);
    if (value < 0)
      value += period;
    if (value >= size)
      value = period - 1 - value;
  } else if (wrap >= ImageClamp::HalfBorder) {
    if ((wrap == ImageClamp::MirrorHalfBorder || wrap == ImageClamp::MirrorBorder) && value < 0)
      value = -1 - value;
    if (value < 0 || value >= size)
      return std::nullopt;
  } else {
    if (wrap == ImageClamp::MirrorOnce && value < 0)
      value = -1 - value;
    value = std::clamp(value, 0.0, double(size - 1));
  }
  return static_cast<uint32_t>(value);
}

inline uint32_t image_selected_mip(double clamped_lod, ImageMipFilter filter) {
  return static_cast<uint32_t>(
      std::floor(clamped_lod + (filter == ImageMipFilter::Point ? 0.5 : 0.0)));
}

// HIP tex1D may view a 2D allocation as a 1D texture. DIM supplies X
// and the sampler supplies zero for the missing Y coordinate.
inline bool image_sample_view_matches(uint32_t dim, uint32_t resource_type) {
  return dim + kImage1d == resource_type || image_sample_has_implicit_y(dim, resource_type);
}

inline uint32_t image_sample_cube_token(double coordinate) {
  return static_cast<uint32_t>(
      std::clamp(std::isnan(coordinate) ? 0.0 : coordinate, 0.0, double(UINT32_MAX)));
}

/// GFX10 cross-face gradient unfolding has not been qualified. Cube array
/// indices do not affect whether the three gradient coordinates share a face.
template <typename Read>
inline bool gfx10_cube_derivatives_use_one_face(ImageDerivativeQuad quad, Read read) {
  const auto face = [&](unsigned lane) {
    return std::min(image_sample_cube_token(read(2, lane)) & 7u, 5u);
  };
  const unsigned origin = face(quad.origin);
  return face(quad.right()) == origin && face(quad.bottom()) == origin;
}

/// Query the quad's sampling LOD without accessing texture memory. Read the
/// entire coordinate quad before writing VDATA, which can alias VADDR.
inline void execute_gfx10_image_get_lod(Wavefront &wf, uint32_t resource_reg, uint32_t sampler_reg,
                                        uint32_t data, const ImageAddressRegisters &vaddr,
                                        ImageFields fields) {
  const uint32_t dst = wf.vgpr_alloc().base + data;
  // GFX10 MIMG permits D16 on sample, gather, load and store operations.
  if (fields.d16 || !image_lod_mask_supported(wf.cu().arch(), fields.dmask)) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  RegisterAccess regs(wf);
  if (!wf.exec() || !wf.vgpr_alloc().contains(dst, std::max(1, std::popcount(fields.dmask & 15u))))
    return;
  if (!fields.dmask) {
    for (uint64_t lanes = wf.exec(); lanes; lanes &= lanes - 1)
      write_image_lod(regs, dst, std::countr_zero(lanes), 0, 0, 0);
    return;
  }
  std::array<uint32_t, 8> descriptor{};
  std::array<uint32_t, 4> sampler_words{};
  const uint32_t words = fields.r128 ? 4 : 8;
  if (!scalar_selector_range_is_backed(wf, resource_reg, words) ||
      !scalar_selector_range_is_backed(wf, sampler_reg, sampler_words.size()))
    return;
  for (uint32_t word_index = 0; word_index < words; ++word_index)
    descriptor[word_index] = read_scalar_selector(wf, resource_reg + word_index);
  for (uint32_t word_index = 0; word_index < sampler_words.size(); ++word_index)
    sampler_words[word_index] = read_scalar_selector(wf, sampler_reg + word_index);
  const ImageResource resource = decode_image_resource(descriptor, wf.cu().arch(), fields.r128);
  const ImageSampler sampler = decode_image_sampler(sampler_words, wf.cu().arch());
  const uint32_t axes = image_derivative_component_count(fields.dim);
  const uint32_t base = wf.vgpr_alloc().base;
  const bool bound = resource.type >= kImage1d && resource.data_format != 0;
  if (bound && !image_supports_resource_type(resource.type, fields.r128, /*sampled=*/true)) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  if (bound &&
      (!image_sample_view_matches(fields.dim, resource.type) || sampler.derivative_adjust ||
       sampler.secondary_lod_bias || sampler.aniso_override || sampler.mip_point_preclamp ||
       sampler.max_aniso_ratio > 4 || sampler.mip_filter > ImageMipFilter::Linear ||
       resource.last_level < resource.base_level)) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    return;
  }
  const std::array<uint32_t, 3> extents =
      fields.unorm || sampler.force_unnormalized
          ? std::array<uint32_t, 3>{1, 1, 1}
          : std::array<uint32_t, 3>{std::max(1u, resource.width >> resource.base_level),
                                    std::max(1u, resource.height >> resource.base_level),
                                    std::max(1u, resource.depth >> resource.base_level)};
  const bool anisotropic =
      ((static_cast<uint32_t>(sampler.min_filter) | static_cast<uint32_t>(sampler.mag_filter)) &
       2u) != 0;
  // Cube and volume footprints use their major axis.
  const uint32_t max_anisotropy =
      anisotropic && fields.dim != kImageDimCube && axes <= 2 ? 1u << sampler.max_aniso_ratio : 0;
  const auto coordinate = [&](unsigned axis, unsigned lane) {
    const uint32_t selector = vaddr[fields.a16 ? axis / 2 : axis];
    const uint32_t reg = base + (selector < wf.num_vgprs() ? selector : 0);
    const uint8_t bytes = image_coordinate_component_bytes(axis, fields.a16);
    const uint32_t bits = regs.read_vgpr(reg, lane, bytes);
    return double(fields.a16 ? util::f16_to_f32(static_cast<uint16_t>(bits >> (16 * (axis % 2))))
                             : std::bit_cast<float>(bits));
  };
  // Validate every live quad before writing VDATA, including when it aliases VADDR.
  if (bound && fields.dim == kImageDimCube) {
    for (unsigned lane = 0; lane < wf.wf_size(); lane += 4)
      if ((wf.exec() & (uint64_t{15} << lane)) &&
          !gfx10_cube_derivatives_use_one_face(ImageDerivativeQuad(lane), coordinate)) {
        wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
        return;
      }
  }
  for (uint32_t quad = 0; quad < wf.wf_size(); quad += 4) {
    if (!(wf.exec() & (uint64_t{15} << quad)))
      continue;
    std::array<double, 3> dx{}, dy{};
    for (uint32_t axis = 0; axis < axes && bound; ++axis) {
      const double origin = coordinate(axis, quad);
      dx[axis] = coordinate(axis, quad + 1) - origin;
      dy[axis] = coordinate(axis, quad + 2) - origin;
    }
    const double raw = bound ? image_gradient_lod(dx, dy, extents, axes, max_anisotropy) +
                                   double(sampler.lod_bias_fixed8) / kImageLodFractionScale
                             : 0;
    const double minimum =
        double(std::max(sampler.min_lod_fixed8, resource.min_lod)) / kImageLodFractionScale;
    const double maximum =
        std::max(minimum, double(sampler.max_lod_fixed8) / kImageLodFractionScale);
    const double clamped = bound && sampler.mip_filter != ImageMipFilter::None
                               ? std::clamp(std::clamp(raw, minimum, maximum), 0.0,
                                            double(resource.last_level - resource.base_level))
                               : 0;
    double selected = image_selected_mip(clamped, sampler.mip_filter);
    if (sampler.mip_filter == ImageMipFilter::Linear) {
      // PERF_MIP snaps the fractional result only for trilinear selection.
      const uint32_t fraction =
          static_cast<uint32_t>((clamped - selected) * kImageLodFractionScale);
      selected += double(image_mip_fraction(fraction, sampler.perf_mip, resource.perf_mod)) /
                  kImageLodFractionScale;
    }
    for (uint32_t lane = quad; lane < quad + 4; ++lane)
      if (wf.exec() & (uint64_t{1} << lane))
        write_image_lod(regs, dst, lane, fields.dmask, static_cast<float>(selected),
                        static_cast<float>(raw));
  }
}

/// @brief Capture the texel footprint before any memory request or destination write.
inline util::Result prepare_image_sample(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg, uint32_t sampler_reg,
                                         ImageAddressRegisters vaddr, ImageFields fields,
                                         ImageSampleMode mode, bool result_in_range = true) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  if (fields.gather && !fields.packed_texels && std::popcount(fields.dmask & 15u) != 1)
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const util::FailureOr<ImageResource> decoded =
      prepare_image_resource(wf, state, resource_reg, fields, result_in_range, /*sampled=*/true);
  if (decoded.failed())
    return util::Result::failure();
  if (fields.gather && !fields.packed_texels)
    state.buffer_components = 4;
  const ImageResource &resource = decoded.value();
  if (!resource.width)
    return util::Result::success();
  const bool gfx10 =
      wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2;
  if (resource.has_unqualified_sample_controls(wf.cu().arch()))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx9 = arch_is_cdna_4_or_lower(wf.cu().arch());
  // The GFX9/10 manuals describe an upper bound, while shader compilers use
  // _CL for minimum LOD. The older hardware behavior has not been qualified.
  if (fields.lod_clamp && (gfx9 || gfx10))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx12 = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t type = gfx9 ? resource.type : fields.dim + kImage1d;
  if (!gfx9 && !image_sample_view_matches(fields.dim, resource.type))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool one_dimensional_view = type != resource.type;
  const ImageCoordinateShape shape = image_coordinate_shape(type, !gfx9 || fields.da);
  const bool has_y = shape.has_y;
  const bool volume = shape.volume;
  const bool gfx11_volume = volume && !gfx12 && has_gfx11_plus_image_operands(wf.cu().arch());
  const bool layered = shape.layered;
  const bool has_slice = shape.has_slice;
  const uint32_t coordinates = shape.coordinates();
  const uint32_t spatial_dimensions = shape.spatial_dimensions();
  const uint32_t filter_dimensions = spatial_dimensions + one_dimensional_view;
  const bool gradients =
      mode == ImageSampleMode::Derivatives || mode == ImageSampleMode::Derivatives16;
  const bool implicit_gradients =
      mode == ImageSampleMode::Implicit || mode == ImageSampleMode::Bias;
  if (gfx9 && fields.a16 && gradients)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const uint32_t gradient_base = fields.offset + fields.lod_bias + fields.compare;
  const uint32_t gradient_stride =
      image_gradient_vector_words(spatial_dimensions, fields.gradient16);
  const uint32_t prefix_words = gradient_base + (gradients ? 2 * gradient_stride : 0);
  const uint16_t prefix_low16_words =
      image_prefix_low16_words(gradient_base, fields.gradient16 ? spatial_dimensions : 0,
                               !gfx9 && fields.a16, fields.lod_bias, fields.offset);
  const uint32_t inputs = coordinates + (mode == ImageSampleMode::Explicit || fields.lod_clamp);
  const uint32_t registers = prefix_words + (fields.a16 ? (inputs + 1) / 2 : inputs);
  if (fields.packed_texels &&
      (fields.d16 || state.image_bc || state.elem_size * fields.packed_texels > 16u))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (fields.gather && (!has_y || volume))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (!resolve_image_coordinates(wf, vaddr, registers))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const bool storage_has_y = has_y || one_dimensional_view;
  const uint32_t pitch = resource.layout_pitch(wf.cu().arch(), fields.dim == kImageDim2d);
  if (!fields.r128 && resource.uses_linear_transfer_pitch() && !state.image_bc &&
      has_gfx11_plus_image_operands(wf.cu().arch()) &&
      !resource.supports_linear_pitch(wf.cu().arch(), fields.dim == kImageDim2d, state.elem_size))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const ImageLayout layout{.width = resource.width,
                           .height = storage_has_y ? resource.height : 1u,
                           .depth = layered || volume ? resource.depth : 1u,
                           .pitch = pitch,
                           .element_bytes = state.elem_size,
                           .swizzle_mode = resource.swizzle_mode,
                           .dimensions = volume          ? 3u
                                         : storage_has_y ? 2u
                                                         : 1u,
                           .mip_levels = resource.max_level + 1};
  ImageMipAddressCache addresses(wf.cu().image_address_config(), layout, resource.base,
                                 state.image_bc.get());
  if (!addresses.get(resource.base_level))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!scalar_selector_range_is_backed(wf, sampler_reg, 4))
    return util::Result::success();
  std::array<uint32_t, 4> sampler_words;
  for (uint32_t word = 0; word < sampler_words.size(); ++word)
    sampler_words[word] = read_scalar_selector(wf, sampler_reg + word);
  const ImageSampler sampler = decode_image_sampler(sampler_words, wf.cu().arch());
  const bool cube = type == kImageCube;
  if (cube && (resource.width != resource.height || resource.depth - resource.base_array < 6))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const bool seamless_cube = cube && !sampler.disable_cube_wrap;
  const bool normalized = !fields.unorm && !sampler.force_unnormalized;
  const bool truncate_coordinates = sampler.truncate_coordinates;
  std::array<ImageClamp, 3> wrap = sampler.clamp;
  const bool gather4h = fields.horizontal && !fields.packed_texels;
  // Match the graphics GATHER4H footprint in image_transfer.h: only border
  // addressing survives, and four horizontal taps start 1.5 texels left of the center.
  if (gather4h)
    for (auto &axis : wrap)
      if (axis != ImageClamp::Border)
        axis = ImageClamp::Edge;
  for (uint32_t axis = 0; axis < filter_dimensions; ++axis)
    if (!gfx10 && wrap[axis] > ImageClamp::MirrorOnce && wrap[axis] != ImageClamp::Border)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  // GFX11/12 captures: unnormalized repeat clamps; mirror reflects once, then clamps.
  // CLR normally sanitizes this combination before constructing the sampler.
  if (!normalized)
    for (ImageClamp &axis : wrap)
      axis = static_cast<ImageClamp>(unnormalized_image_clamp(static_cast<uint32_t>(axis)));
  const ImageXyFilter mag_filter = sampler.mag_filter;
  const ImageXyFilter min_filter = sampler.min_filter;
  const ImageMipFilter mip_filter = fields.gather && sampler.mip_filter == ImageMipFilter::Linear
                                        ? ImageMipFilter::Point
                                        : sampler.mip_filter;
  if (mag_filter > ImageXyFilter::Linear || min_filter > ImageXyFilter::Linear ||
      (!gfx10 && !gfx11_volume && mag_filter != min_filter) ||
      mip_filter > ImageMipFilter::Linear || sampler.border_color_type > (gfx10 ? 2u : 0u) ||
      sampler.filter_mode || sampler.max_aniso_ratio || sampler.force_degamma ||
      sampler.mc_coord_trunc || sampler.skip_degamma || sampler.z_filter ||
      sampler.secondary_lod_bias || sampler.mip_point_preclamp || sampler.filter_precision_fix ||
      sampler.compat_mode ||
      (sampler.blend_prt && !has_gfx11_plus_image_operands(wf.cu().arch())) ||
      has_unimplemented_image_sampler_controls(sampler) ||
      (resource.perf_mod && ((!fields.gather && sampler.perf_mip) || (volume && sampler.perf_z))))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!fields.gather &&
      (mag_filter == ImageXyFilter::Linear || min_filter == ImageXyFilter::Linear ||
       mip_filter == ImageMipFilter::Linear) &&
      (state.decoded_buffer_format.number == BufferNumberFormat::Uint ||
       state.decoded_buffer_format.number == BufferNumberFormat::Sint))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.image_status && !gfx12 && has_gfx11_plus_image_operands(wf.cu().arch()))
    state.image_status->blend_prt = sampler.blend_prt;
  auto access = std::make_unique<ImageSampleAccess>();
  access->gather = fields.gather;
  access->border_color = sampler.border_color_type;
  access->horizontal = fields.horizontal;
  access->packed_row = fields.packed_texels != 0;
  if (fields.compare) {
    access->depth_compare = std::make_unique<ImageSampleAccess::DepthCompare>();
    access->depth_compare->function = sampler.compare_function;
  }
  const bool multi_tap = fields.gather || mag_filter == ImageXyFilter::Linear ||
                         min_filter == ImageXyFilter::Linear ||
                         mip_filter == ImageMipFilter::Linear;
  const uint32_t spatial_taps = fields.packed_texels ? fields.packed_texels
                                : multi_tap          ? (1u << filter_dimensions)
                                                     : 1;
  const uint32_t mip_taps = mip_filter == ImageMipFilter::Linear && !fields.gather ? 2 : 1;
  if (has_gfx11_plus_image_operands(wf.cu().arch()) && image_is_1d_or_2d_sample(fields.dim) &&
      (spatial_taps == 2 || spatial_taps == 4) &&
      state.decoded_buffer_format.number == BufferNumberFormat::Float &&
      state.decoded_buffer_format.widths[0] == 32)
    access->fp32_filter_taps = spatial_taps;
  const uint32_t primary_taps = spatial_taps * mip_taps;
  // At a cube corner, the missing bilinear texel is the average of the
  // three incident faces. Gather returns one unfiltered value per tap.
  const bool corner_blend =
      seamless_cube && !fields.gather &&
      (mag_filter == ImageXyFilter::Linear || min_filter == ImageXyFilter::Linear);
  access->taps.resize(primary_taps * (corner_blend ? 3 : 1));
  RegisterAccess regs(wf);
  const auto read = [&](uint32_t component, uint32_t lane) {
    const uint32_t bits = regs.read_vgpr(
        wf.vgpr_alloc().base + vaddr[prefix_words + (fields.a16 ? component / 2 : component)], lane,
        image_coordinate_component_bytes(component, !gfx9 && fields.a16));
    return double(fields.a16 ? util::f16_to_f32((bits >> (16 * (component % 2))) & 0xffffu)
                             : std::bit_cast<float>(bits));
  };
  double coarse_lod = 0;
  uint32_t coarse_quad = UINT32_MAX;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(state.exec_mask & (uint64_t{1} << lane)))
      continue;
    const auto read_prefix = [&](unsigned index) {
      return regs.read_vgpr(wf.vgpr_alloc().base + vaddr[index], lane,
                            image_prefix_word_bytes(index, prefix_low16_words));
    };
    const uint32_t offset = fields.offset ? read_prefix(0) : 0;
    const uint32_t bias_bits = fields.lod_bias ? read_prefix(fields.offset) : 0;
    const double instruction_bias = fields.a16 ? util::f16_to_f32(static_cast<uint16_t>(bias_bits))
                                               : std::bit_cast<float>(bias_bits);
    if (fields.compare)
      access->depth_compare->reference[lane] =
          std::bit_cast<float>(read_prefix(fields.offset + fields.lod_bias));
    double lod = mode == ImageSampleMode::Explicit ? read(coordinates, lane) : 0;
    if (gradients || (implicit_gradients &&
                      linear_image_needs_implicit_lod(sampler_words[2], resource.base_level,
                                                      resource.last_level, fields.gather))) {
      std::array<double, 3> dx{}, dy{};
      const std::array<uint32_t, 3> extents{std::max(1u, resource.width >> resource.base_level),
                                            std::max(1u, resource.height >> resource.base_level),
                                            std::max(1u, resource.depth >> resource.base_level)};
      if (implicit_gradients) {
        const ImageDerivativeQuad quad(lane);
        const uint64_t quad_mask = uint64_t{15} << quad.origin;
        if (!has_gfx11_plus_image_operands(wf.cu().arch()) &&
            (state.exec_mask & quad_mask) != quad_mask)
          return fail(InstructionExecutionError::UnimplementedInstruction);
        if (gfx10 && cube && !gfx10_cube_derivatives_use_one_face(quad, read))
          return fail(InstructionExecutionError::UnimplementedInstruction);
        for (uint32_t axis = 0; axis < spatial_dimensions; ++axis) {
          dx[axis] = read(axis, quad.right()) - read(axis, quad.origin);
          dy[axis] = read(axis, quad.bottom()) - read(axis, quad.origin);
        }
      } else {
        for (uint32_t axis = 0; axis < spatial_dimensions; ++axis) {
          const auto gradient = [&](uint32_t direction) -> double {
            const uint32_t word = read_prefix(gradient_base + direction * gradient_stride +
                                              (fields.gradient16 ? axis / 2 : axis));
            if (fields.gradient16)
              return util::f16_to_f32(static_cast<uint16_t>(word >> (16 * (axis % 2))));
            return std::bit_cast<float>(word);
          };
          dx[axis] = gradient(0);
          dy[axis] = gradient(1);
        }
      }
      lod = image_gradient_lod(dx, dy, normalized ? extents : std::array<uint32_t, 3>{1, 1, 1},
                               spatial_dimensions);
      if (fields.coarse_derivatives) {
        // CD supplies one LOD per quad. A lane selection for disagreeing
        // gradients has not been qualified on GFX10.
        const uint32_t quad = lane / 4;
        if (quad == coarse_quad && lod != coarse_lod)
          return fail(InstructionExecutionError::UnimplementedInstruction);
        coarse_quad = quad;
        coarse_lod = lod;
      }
    }
    // Shader bias has its own fixed-point rounding before footprint addition on GFX11/12.
    lod += gfx9 || gfx10 ? instruction_bias : image_sample_round_even8(instruction_bias);
    if (std::isnan(lod))
      lod = 0;
    else if (std::isinf(lod))
      lod = std::copysign(double(std::numeric_limits<float>::max()), lod);
    const double minimum_lod =
        double(std::max(sampler.min_lod_fixed8, resource.min_lod)) / kImageLodFractionScale;
    const double maximum_lod = double(sampler.max_lod_fixed8) / kImageLodFractionScale;
    const int32_t bias = sampler.lod_bias_fixed8;
    lod = image_sample_round_even8(lod) + double(bias) / kImageLodFractionScale;
    if (fields.lod_clamp) {
      const auto clamped = image_shader_min_lod(lod, read(coordinates, lane));
      if (!clamped)
        return fail(InstructionExecutionError::UnsupportedOperandValue);
      lod = *clamped;
    }
    lod = std::clamp(lod, minimum_lod, std::max(minimum_lod, maximum_lod));
    const bool minifying = lod > 0;
    const double requested_lod = resource.base_level + lod;
    lod = mip_filter != ImageMipFilter::None
              ? std::clamp(lod, 0.0, double(resource.last_level - resource.base_level))
              : 0;
    if (state.texel_status) {
      // The GFX10 model compares the clamped LOD (base level with mip filtering
      // disabled) and reports its integer part before point-mip rounding. Hardware
      // qualification of this status field is still pending on GFX10.
      const double absolute_lod = lod + resource.base_level;
      state.texel_status->lod[lane] = static_cast<uint8_t>(std::floor(absolute_lod));
      if (fields.lwe && absolute_lod * kImageLodFractionScale < resource.min_lod_warn)
        state.texel_status->lod_warnings |= uint64_t{1} << lane;
    }
    const uint32_t first_level = resource.base_level + image_selected_mip(lod, mip_filter);
    if (state.image_status)
      state.image_status->record_sample(lane, mip_filter != ImageMipFilter::None, requested_lod,
                                        first_level);
    const double mip_fraction =
        mip_filter == ImageMipFilter::Linear ? image_sample_fixed8(lod - std::floor(lod)) : 0;
    if (gfx12 && mip_fraction != 0)
      access->mip_blend_mask |= uint64_t{15} << (lane & ~3u);
    const bool linear =
        fields.gather || (minifying ? min_filter : mag_filter) == ImageXyFilter::Linear;
    double layer = layered && has_slice ? read(has_y ? 2 : 1, lane) : 0;
    if (std::isnan(layer))
      layer = 0;
    const double lower_layer = std::floor(layer);
    const double layer_fraction = layer - lower_layer;
    const double rounded_layer =
        lower_layer +
        (layer_fraction > 0.5 || (layer_fraction == 0.5 && std::fmod(lower_layer, 2.0) != 0));
    uint32_t array_layer =
        resource.base_array +
        static_cast<uint32_t>(
            std::clamp(rounded_layer, 0.0, double(resource.depth - resource.base_array - 1)));
    uint32_t face = 0, cube_base = 0;
    if (cube) {
      // Sampler cube-array tokens use three low bits for the face. Raw
      // IMAGE_LOAD/STORE instead take a linear face index.
      const uint32_t token = image_sample_cube_token(layer);
      face = std::min(token & 7u, 5u);
      cube_base = resource.base_array +
                  6 * std::min(token >> 3, (resource.depth - resource.base_array - 6) / 6);
      array_layer = cube_base + face;
    }
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
        if (cube)
          value -= 1.0;
        value *= normalized ? sizes[axis] : 1;
        const uint32_t encoded_offset = (offset >> (axis * 8)) & 0x3fu;
        value += static_cast<int32_t>(encoded_offset ^ 32u) - 32;
        if (wrap[axis] == ImageClamp::HalfBorder || wrap[axis] == ImageClamp::MirrorHalfBorder) {
          if (wrap[axis] == ImageClamp::MirrorHalfBorder)
            value = std::abs(value);
          value = std::clamp(value, 0.0, double(sizes[axis]));
        }
        const double center = gather4h ? (axis == 0 ? 1.5 : 0.0) : (linear ? 0.5 : 0.0);
        const double position = value - center;
        origins[axis] =
            std::floor(!linear && !truncate_coordinates ? image_sample_fixed8(position) : position);
        fractions[axis] = linear ? image_sample_round_even8(position - origins[axis]) : 0;
        // Duplicate texels collapse before interpolation. image_sample_values
        // returns a lone weight-1 tap's raw payload to preserve signed zero;
        // GFX12 fractional-mip quads explicitly disable that shortcut.
        const std::optional<uint32_t> lower =
            image_sample_coordinate(origins[axis], sizes[axis], wrap[axis]);
        if (!seamless_cube && lower &&
            lower == image_sample_coordinate(origins[axis] + 1, sizes[axis], wrap[axis]))
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
          const std::optional<uint32_t> selected = image_sample_coordinate(
              origins[axis] + (fields.horizontal ? (axis == 0 ? tap : 0) : upper), sizes[axis],
              wrap[axis]);
          border |= !selected;
          coordinate[axis] = selected.value_or(0);
        }
        if (fields.gather)
          weight = 1;
        destination.weights[lane] = weight;
        if (weight == 0)
          continue;
        uint32_t z = layered ? array_layer : volume ? coordinate[2] : 0;
        if (seamless_cube) {
          const double x = origins[0] + (fields.horizontal ? tap : (tap & 1u));
          const double y = origins[1] + (fields.horizontal ? 0 : ((tap >> 1) & 1u));
          const double edge = sizes[0] - 1;
          const double cx = std::clamp(x, 0.0, edge), cy = std::clamp(y, 0.0, edge);
          if (corner_blend && x != cx && y != cy) {
            destination.weights[lane] = weight / 3;
            for (uint32_t i = 1; i <= 2; ++i) {
              const auto texel = image_cube_texel(face, i == 1 ? x : cx, i == 1 ? cy : y, sizes[0]);
              auto &extra = access->taps[i * primary_taps + mip_tap * spatial_taps + tap];
              extra.weights[lane] = weight / 3;
              extra.addresses[lane] = addr_calc::buffer_virtual_address(
                  resource.base +
                  addresses.offset(*addressing, texel.x, texel.y, cube_base + texel.face, lane,
                                   i * primary_taps + mip_tap * spatial_taps + tap));
              extra.lane_mask |= uint64_t{1} << lane;
            }
            coordinate = {static_cast<uint32_t>(cx), static_cast<uint32_t>(cy), 0};
          } else {
            const auto texel = image_cube_texel(face, x, y, sizes[0]);
            coordinate = {texel.x, texel.y, 0};
            z = cube_base + texel.face;
          }
        } else if (border) {
          if (fields.packed_texels && sampler.border_color_type)
            return fail(InstructionExecutionError::UnimplementedInstruction);
          destination.border_mask |= uint64_t{1} << lane;
          continue;
        }
        if (!addresses.contains(*addressing, coordinate[0], storage_has_y ? coordinate[1] : 0, z))
          return fail(InstructionExecutionError::UnsupportedOperandValue);
        const uint64_t offset =
            addresses.offset(*addressing, coordinate[0], storage_has_y ? coordinate[1] : 0, z, lane,
                             mip_tap * spatial_taps + tap);
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
