// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_sample.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/graphics/image_address.h"
#include "rocjitsu/vm/graphics/image_cube.h"
#include "rocjitsu/vm/graphics/image_filter.h"
#include "rocjitsu/vm/graphics/image_msaa.h"
#include "rocjitsu/vm/graphics/image_volume.h"
#include "util/data_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {

enum class ImageTransferMode {
  Default,
  Mip,
  Packed,
  PackedSigned,
  MipPacked,
  MipPackedSigned,
  Msaa
};

using ImageLodResults = std::array<std::array<float, 2>, 64>;

inline bool uses_native_2d_sampler(const Wavefront &wf, uint32_t resource, uint32_t sampler,
                                   uint32_t dim, bool unorm) {
  const RegisterAccess registers(wf);
  return uses_native_2d_sampler(
      wf.cu().arch(), resource, sampler, dim, unorm,
      [&](uint32_t index) { return registers.scalar_control_value(index).value_or(0); },
      [&](uint32_t index, uint32_t count) {
        return scalar_selector_range_is_backed(wf, index, count);
      });
}

inline bool use_linear_image_path(Wavefront &wf, uint32_t resource, uint32_t sampler, uint32_t dim,
                                  bool unorm, bool sample) {
  const RegisterAccess registers(wf);
  return !(sample && uses_native_2d_sampler(wf, resource, sampler, dim, unorm)) &&
         use_linear_image_path(
             wf.cu().arch(), resource, sampler, dim, sample,
             [&](uint32_t index) { return registers.scalar_control_value(index).value_or(0); },
             [&](uint32_t index, uint32_t count) {
               return scalar_selector_range_is_backed(wf, index, count);
             });
}

/// Prepare GFX11/12 image transfers and sampling.
inline util::Result
prepare_image_transfer(Wavefront &wf, VectorMemState &state, uint32_t resource, uint32_t data,
                       ImageAddressRegisters coords, ImageFields controls, uint32_t sampler = ~0u,
                       ImageSampleMode sample_mode = ImageSampleMode::Implicit,
                       ImageLodResults *queried_lods = nullptr,
                       ImageTransferMode transfer_mode = ImageTransferMode::Default) {
  const uint32_t dim = controls.dim;
  uint32_t &mask = controls.dmask;
  const bool a16 = controls.a16, d16 = controls.d16;
  const auto unsupported = [&] {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return util::Result::failure();
  };
  const auto arch = wf.cu().arch();
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool query = queried_lods != nullptr;
  const bool atomic = state.atomic_op != AtomicOp::NONE;
  state.check_texel_residency = !state.is_load && !atomic;
  const bool mip_transfer = transfer_mode == ImageTransferMode::Mip ||
                            transfer_mode == ImageTransferMode::MipPacked ||
                            transfer_mode == ImageTransferMode::MipPackedSigned;
  const bool quad_fragments = transfer_mode == ImageTransferMode::Msaa;
  const bool gather = controls.gather || quad_fragments;
  const bool packed = transfer_mode != ImageTransferMode::Default &&
                      transfer_mode != ImageTransferMode::Mip && !quad_fragments;
  const bool packed_signed = transfer_mode == ImageTransferMode::PackedSigned ||
                             transfer_mode == ImageTransferMode::MipPackedSigned;
  // An unbound descriptor suppresses memory access but still requires a legal atomic payload.
  const unsigned atomic_bytes =
      atomic ? image_atomic_texel_bytes(state.atomic_op, mask,
                                        atomic_is_floating_point(state.atomic_op))
             : 0;
  if ((!gfx12 && arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
      (atomic && (!controls.unorm || d16 || !atomic_bytes)) || (packed && d16) || (mask & ~15u) ||
      (query && (d16 || (mask & ~3u))) || (gather && std::popcount(mask) != 1))
    return unsupported();
  const uint32_t words = controls.r128 ? 4 : 8;
  const bool dropped = mask == 0;
  // Allocation includes the PRT register even when the fetch drops or a LOD
  // query cannot NACK. LWE and TFE share one trailing DWORD.
  const bool feedback = state.is_load && (controls.tfe || (sampler != ~0u && controls.lwe));
  const uint32_t components = !mask ? 1 : gather ? 4 : std::popcount(mask);
  // DMASK includes both the replacement and comparison for CMPSWAP. Only
  // the previous texel and optional feedback belong to its returned tuple.
  const uint32_t result_components =
      state.atomic_op == AtomicOp::CMPSWAP ? components / 2 : components;
  // Even when a transfer has no memory lanes, pipeline admission checks the
  // atomic element width. A bound descriptor supplies its actual width below.
  state.elem_size = atomic ? atomic_bytes : 1u;
  state.num_elems = 1;
  const uint32_t data_words = d16 ? (result_components + 1) / 2 : result_components;
  const uint32_t tuple_words = data_words + uint32_t(feedback);
  const bool validate_tuple = state.is_load;
  if (validate_tuple && (data >= wf.num_vgprs() || tuple_words > wf.num_vgprs() - data)) {
    // An out-of-range destination suppresses the entire transaction, including feedback.
    state.exec_mask = state.lane_mask = 0;
    state.wf_size = wf.wf_size();
    return util::Result::success();
  }
  if (dropped && !(sampler != ~0u && controls.lwe)) {
    state.dst_reg_base = wf.vgpr_alloc().base + data;
    state.buffer_components = 1;
    state.buffer_d16 = d16;
    state.wf_size = wf.wf_size();
    state.exec_mask = wf.exec();
    state.lane_mask = 0;
    return util::Result::success();
  }
  if (dropped)
    mask = 1; // LWE still needs the per-lane requested LOD.
  std::array<uint32_t, 8> resource_words{};
  for (uint32_t i = 0; i < words; ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return util::Result::failure();
  for (uint32_t i = 0; i < words; ++i)
    resource_words[i] = read_scalar_selector(wf, resource + i);
  // An all-zero descriptor is an unbound image. GFX11/12 return zero
  // without fetching coordinates, sampler state, or image memory.
  const auto empty_transfer = [&]() {
    // Compare-swap consumes replacement and comparison words but returns only
    // the previous texel. An unbound descriptor has no PRT status to write.
    state.buffer_components =
        gather ? 4 : std::popcount(mask) / (state.atomic_op == AtomicOp::CMPSWAP ? 2 : 1);
    state.buffer_d16 = d16;
    state.wf_size = wf.wf_size();
    state.exec_mask = wf.exec();
    state.lane_mask = 0;
    state.dst_reg_base = wf.vgpr_alloc().base + data;
    return util::Result::success();
  };
  if (image_descriptor_is_unbound(arch, resource_words[1], resource_words[3]))
    return empty_transfer();
  const ImageResource descriptor = decode_image_resource(resource_words, arch, controls.r128);
  const uint32_t type = descriptor.type;
  if (controls.r128 && type != kImage1d && type != kImage2d && type != kImage2dMsaa)
    return unsupported();
  if (!state.is_load && !atomic && !controls.unorm)
    return unsupported();
  const bool compressed = !gfx12 && (resource_words[6] & (1u << 21)) &&
                          ((resource_words[6] >> 24) || resource_words[7]);
  const bool resident_mip = descriptor.min_lod && !query && sampler != ~0u && !controls.compare &&
                            !controls.offset && !controls.lod_clamp &&
                            scalar_selector_range_is_backed(wf, sampler, 4) &&
                            descriptor.supports_resident_mip(
                                dim, sample_mode, controls.gather,
                                RegisterAccess(wf).scalar_control_value(sampler + 2).value_or(0));
  if (descriptor.linked_resource || (!query && descriptor.uav3d) ||
      (sampler != ~0u && !query &&
       descriptor.has_unqualified_sample_controls(arch, resident_mip)) ||
      (!query && descriptor.compressed && !compressed) ||
      !image_dim_has_resource_layer(dim, type)) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    return util::Result::failure();
  }
  const uint32_t warn_lod = rdna_image_lod_warning_threshold(arch, resource_words[5]);
  if (!query && (state.is_load || atomic) &&
      (warn_lod || controls.tfe || controls.lwe || image_may_have_sparse_pages(wf))) {
    state.image_status = std::make_unique<ImageFeedback>();
    state.image_status->warning_threshold = warn_lod;
    state.image_status->tfe = controls.tfe && state.is_load;
    state.image_status->lwe = sampler != ~0u && controls.lwe;
  }
  const bool multisample = type == kImage2dMsaa || type == kImage2dMsaaArray;
  const bool is_array = type == kImageCube || type == kImage1dArray || type == kImage2dArray ||
                        type == kImage2dMsaaArray;
  const uint32_t msaa_dim = type == kImage2dMsaa ? kImageDim2dMsaa : kImageDim2dMsaaArray;
  if ((multisample && (dim != msaa_dim || sampler != ~0u || mip_transfer)) ||
      (!multisample && (dim > kImageDim2dArray || quad_fragments)))
    return unsupported();
  const bool volume = type == kImage3d;
  const bool compute_feedback =
      !dropped && !controls.r128 && (controls.tfe || controls.lwe || warn_lod) &&
      !controls.gather && !controls.lod_clamp && !query && sampler != ~0u &&
      scalar_selector_range_is_backed(wf, sampler, 4) &&
      use_linear_image_path(
          arch, resource, sampler, dim, true,
          [&](uint32_t n) { return RegisterAccess(wf).scalar_control_value(n).value_or(0); },
          [&](uint32_t n, uint32_t count) { return scalar_selector_range_is_backed(wf, n, count); },
          true) &&
      compute_filtered_image_feedback(
          arch, dim, resource_words[3], resource_words[1],
          RegisterAccess(wf).scalar_control_value(sampler + 2).value_or(0));
  if ((volume || compute_feedback) && sampler != ~0u && !query) {
    if (controls.compare || controls.offset || controls.lod_clamp || controls.gather ||
        sample_mode == ImageSampleMode::Bias || sample_mode == ImageSampleMode::Derivatives16)
      return unsupported();
    // The transfer owns feedback allocation and status. The ordinary sampler
    // prepares resident texels without requesting a second status word.
    ImageFields fields = controls;
    fields.da = true;
    fields.tfe = fields.lwe = false;
    state.dst_reg_base = wf.vgpr_alloc().base + data;
    const auto prepared =
        prepare_image_sample(wf, state, resource, sampler, coords, fields, sample_mode);
    if (prepared.succeeded() && dropped) {
      state.image_sample.reset();
      state.lane_mask = state.buffer_selectors = 0;
      state.buffer_components = 1;
    }
    return prepared;
  }

  // Non-sampling 2D-array instructions also address 3D resources: their third
  // coordinate is Z, with the resource's volume layout and depth bounds.
  if ((dim == kImageDim3d && !volume) ||
      (volume && ((dim != kImageDim3d && dim != kImageDim2dArray) || sampler != ~0u ||
                  (resource_words[5] & 16))))
    return unsupported();
  const uint32_t swizzle = descriptor.swizzle_mode;
  uint32_t width = descriptor.width;
  uint32_t height = descriptor.height;
  const uint32_t image_format = descriptor.data_format;
  const auto bc = decode_image_block_format(image_format, false);
  const uint32_t unorm_format = image_srgb_storage_format(image_format, false);
  const uint32_t format = bc ? (bc->scalar() ? (bc->channels() == 1 ? 22 : 50) : 42) : unorm_format;
  state.image_srgb = (unorm_format != image_format || (bc && bc->srgb())) && !packed;
  state.image_bc_format = bc;
  if (bc && (!state.is_load || atomic || packed || volume))
    return unsupported();
  const auto decoded = decode_buffer_format(format);
  if (decoded.failed())
    return unsupported();
  state.decoded_buffer_format = decoded.value();
  const uint32_t bytes = bc ? bc->bytes() : state.decoded_buffer_format.byte_size();
  if (packed) {
    // PCK ignores format conversion and descriptor channel selectors. DMASK
    // selects raw DWORDs; a sub-DWORD texel occupies one extended word.
    auto &raw = state.decoded_buffer_format;
    for (uint32_t i = 0; i < 4; ++i)
      raw.widths[i] = bytes > i * 4 ? std::min(4u, bytes - i * 4) * 8 : 0;
    raw.number = packed_signed ? BufferNumberFormat::Sint : BufferNumberFormat::Uint;
  }
  if (atomic && (bytes != atomic_bytes || compressed))
    return unsupported();
  const uint32_t fragment_log2 = (resource_words[3] >> 16) & 15;
  if (multisample && (gfx12 || compressed || bc || !fragment_log2 || fragment_log2 > 3 ||
                      !std::has_single_bit(bytes) || bytes > 16 || (resource_words[3] & 0xf000) ||
                      (swizzle != 24 && swizzle != 27 && swizzle != 28 && swizzle != 31)))
    return unsupported();
  const uint32_t max_level = multisample ? 0 : descriptor.max_level;
  const auto mip_range = rdna_image_mip_range(arch, resource_words[1], resource_words[3]);
  const uint32_t first_level = mip_range.first;
  const uint32_t last_level = multisample ? 0 : mip_range.last;
  const uint32_t perf_mod = descriptor.perf_mod;
  const double resident_min_lod = descriptor.min_lod / 256.0;
  const bool sample = sampler != ~0u;
  const bool one_dimensional = dim == kImageDim1d || dim == kImageDim1dArray;
  const bool implicit_y = sample && image_sample_has_implicit_y(dim, type);
  const uint32_t spatial_components = one_dimensional ? 1 : volume ? 3 : 2;
  const bool layer_coordinate =
      !volume && (dim == kImageDimCube || dim == kImageDim1dArray || dim == kImageDim2dArray);
  state.image_sampling = sample;
  const uint32_t first_layer = descriptor.base_array;
  const uint32_t last_layer = is_array ? descriptor.depth - 1 : 0;
  if ((type != kImage1d && type != kImage2d && !is_array && !volume && !multisample) ||
      (one_dimensional && !implicit_y && type != kImage1d && type != kImage1dArray) ||
      ((type == kImage1d || type == kImage1dArray) && (height != 1 || (!query && swizzle))) ||
      (type == kImage1dArray && dim != kImageDim1d && dim != kImageDim1dArray) ||
      (!is_array && !volume && (resource_words[4] >> 16)) || (!query && !bytes) ||
      (is_array &&
       (first_layer > last_layer || (resource_words[4] & (gfx12 ? 0xc000c000u : 0xe000e000u)))) ||
      first_level > last_level || last_level > max_level ||
      (!query && max_level && !is_array && !volume && resource_words[4] &&
       !(type == kImage2d && !swizzle && !bc)) ||
      (!state.is_load && state.image_srgb && !atomic))
    return unsupported();
  if (dim == kImageDimCube && (width != height || first_layer % 6 || last_layer < first_layer ||
                               last_layer - first_layer < 5))
    return unsupported();
  const uint32_t resource_width = width, resource_height = height;
  uint32_t storage_width = resource_width;
  if (!query && !controls.r128 && !bc && descriptor.uses_linear_transfer_pitch()) {
    const uint32_t pitch = descriptor.layout_pitch(arch, sample && dim == kImageDim2d);
    if (!descriptor.supports_linear_pitch(arch, sample && dim == kImageDim2d, bytes)) {
      if (sample)
        return unsupported();
      wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
      return util::Result::failure();
    }
    if (pitch)
      storage_width = pitch;
  }
  const auto make_mip_layout = [&](uint32_t level) {
    auto layout =
        image_mip_layout(gfx12, swizzle, bytes, bc ? (resource_width + 3) / 4 : storage_width,
                         bc ? (resource_height + 3) / 4 : resource_height, max_level + 1, level);
    if (layout && (bc || storage_width != resource_width)) {
      layout->width = std::max(1u, resource_width >> level);
      layout->height = std::max(1u, resource_height >> level);
    }
    return layout;
  };
  const uint32_t resource_depth = volume ? descriptor.depth : 1;
  const auto volume_mip = volume
                              ? image_volume_mip_layout(gfx12, swizzle, bytes, width, height,
                                                        resource_depth, max_level + 1, first_level)
                              : std::optional<ImageVolumeMipLayout>{};
  // Queries use descriptor extents without requiring a supported memory layout.
  const auto mip =
      (query || multisample)
          ? std::optional<ImageMipLayout>{{.width = std::max(1u, width >> first_level),
                                           .height = std::max(1u, height >> first_level)}}
      : volume ? (volume_mip ? std::optional{volume_mip->plane} : std::nullopt)
               : make_mip_layout(first_level);
  if (!mip)
    return unsupported();
  // Every lane and filter tap shares the descriptor's mip layouts. Compute
  // only the levels selected by this instruction, once per level.
  std::array<std::optional<ImageMipLayout>, 32> mip_layouts{};
  mip_layouts[first_level] = mip;
  const auto get_mip_layout = [&](uint32_t level) -> const std::optional<ImageMipLayout> & {
    auto &layout = mip_layouts[level];
    if (!layout)
      layout = make_mip_layout(level);
    return layout;
  };
  width = mip->width;
  height = mip->height;
  bool normalized = true, seamless_cube = false, truncate_coordinates = false;
  uint32_t min_filter = 0, mag_filter = 0, mip_filter = 0, max_anisotropy = 1;
  bool needs_gradients = false;
  uint32_t aniso_threshold = 0, aniso_bias = 0, perf_mip = 0;
  double min_lod = 0, max_lod = 0, lod_bias = 0;
  const bool g16 = sample_mode == ImageSampleMode::Derivatives16;
  const bool derivatives = g16 || sample_mode == ImageSampleMode::Derivatives;
  const uint32_t bias_offset = controls.offset;
  const uint32_t compare_offset = bias_offset + (sample_mode == ImageSampleMode::Bias);
  const uint32_t gradient_offset = compare_offset + controls.compare;
  uint32_t coordinate_offset = gradient_offset;
  uint32_t wrap_x = 2, wrap_y = 2;
  if (sample) {
    // RADV's blit shaders use array coordinates for single-layer 2D resources
    // as well. Their descriptor still bounds the selected layer to zero.
    if ((dim == kImageDim2dArray && type != kImage2d && type != kImage2dArray) || !state.is_load ||
        (controls.gather && one_dimensional))
      return unsupported();
    if (resident_min_lod > last_level || !scalar_selector_range_is_backed(wf, sampler, 4))
      return empty_transfer();
    std::array<uint32_t, 4> sampler_words{};
    for (uint32_t i = 0; i < sampler_words.size(); ++i)
      sampler_words[i] = read_scalar_selector(wf, sampler + i);
    const ImageSampler decoded_sampler = decode_image_sampler(sampler_words, arch);
    if (has_unimplemented_image_sampler_controls(decoded_sampler)) {
      wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
      return util::Result::failure();
    }
    if (state.image_status && !gfx12)
      state.image_status->blend_prt = decoded_sampler.blend_prt;
    wrap_x = static_cast<uint32_t>(decoded_sampler.clamp[0]);
    wrap_y = one_dimensional && !implicit_y ? 2 : static_cast<uint32_t>(decoded_sampler.clamp[1]);
    if (controls.horizontal) {
      // 4H keeps border addressing but treats every other wrap mode as edge clamp.
      wrap_x = wrap_x == 6 ? 6 : 2;
      wrap_y = wrap_y == 6 ? 6 : 2;
    }
    const bool fixed_level = graphics_sample_has_fixed_lod(
        query, controls.lwe, sampler_words[2],
        uses_native_2d_sampler(wf, resource, sampler, dim, controls.unorm));
    needs_gradients = graphics_image_needs_gradients(query, fixed_level ? 0u : max_level, dim,
                                                     controls.gather, sampler_words[2]);
    mag_filter = static_cast<uint32_t>(decoded_sampler.mag_filter);
    min_filter = static_cast<uint32_t>(decoded_sampler.min_filter);
    // Cube sampling uses the major footprint without anisotropic taps.
    // Retain the sampler's LOD bias and mip performance controls below.
    if (dim == kImageDimCube) {
      mag_filter &= 1;
      min_filter &= 1;
    }
    mip_filter = static_cast<uint32_t>(decoded_sampler.mip_filter);
    const uint32_t ratio = decoded_sampler.max_aniso_ratio;
    if (ratio > 4)
      return unsupported();
    max_anisotropy = 1u << ratio;
    aniso_threshold = decoded_sampler.aniso_threshold;
    aniso_bias = decoded_sampler.aniso_bias;
    perf_mip = decoded_sampler.perf_mip;
    const auto supported_wrap = [](uint32_t wrap) { return wrap <= 3 || wrap == 6; };
    if (!supported_wrap(wrap_x) || !supported_wrap(wrap_y) || decoded_sampler.filter_mode ||
        decoded_sampler.mc_coord_trunc || decoded_sampler.force_degamma ||
        decoded_sampler.z_filter || mip_filter > 2)
      return unsupported();
    seamless_cube = dim == kImageDimCube && !decoded_sampler.disable_cube_wrap;
    if (seamless_cube)
      wrap_x = wrap_y = 2;
    normalized = !controls.unorm && !decoded_sampler.force_unnormalized;
    if (!normalized) {
      wrap_x = unnormalized_image_clamp(wrap_x);
      wrap_y = unnormalized_image_clamp(wrap_y);
    }
    truncate_coordinates = decoded_sampler.truncate_coordinates;
    if (dim == kImageDimCube && !normalized)
      return unsupported();
    state.image_srgb &= !decoded_sampler.skip_degamma;
    min_lod = decoded_sampler.min_lod_fixed8 / 256.0;
    max_lod = decoded_sampler.max_lod_fixed8 / 256.0;
    lod_bias =
        decoded_sampler.lod_bias_fixed8 / 256.0 +
        ((decoded_sampler.secondary_lod_bias * int32_t(kImagePerfScales[perf_mod])) >> 4) / 16.0;
    // Eight-/ten-bit UNORM, eight-bit sRGB and one-/two-/four-component float filtering.
    const bool filterable = native_filterable_format(format);
    if (min_lod > max_lod || (!dropped && !query && !controls.gather &&
                              (min_filter || mag_filter || mip_filter == 2) && !filterable))
      return unsupported();
    if (sample_mode == ImageSampleMode::Bias)
      coordinate_offset = compare_offset + controls.compare;
    else if (derivatives)
      coordinate_offset = gradient_offset + (g16 ? 2 : 2 * spatial_components);
    if (controls.gather) {
      // Gather selects one mip and returns a footprint without filtering.
      min_filter = mag_filter = 1;
      mip_filter = mip_filter ? 1 : 0;
    }
    if (!query && (min_filter || mag_filter || mip_filter == 2 || wrap_x == 6 || wrap_y == 6 ||
                   controls.compare)) {
      if (decoded_sampler.border_color_type == 3)
        return unsupported(); // Custom border-color tables.
      state.graphics_sample = std::make_unique<GraphicsSampleAccess>();
      state.graphics_sample->tap_count = mip_filter == 2 ? 8 : (min_filter || mag_filter ? 4 : 1);
      state.graphics_sample->texels_per_tap = seamless_cube ? 3 : 1;
      state.graphics_sample->tap_count *= state.graphics_sample->texels_per_tap;
      state.graphics_sample->taps_per_filter = state.graphics_sample->tap_count;
      state.graphics_sample->taps.resize(state.graphics_sample->tap_count);
      state.graphics_sample->border_color = decoded_sampler.border_color_type;
      state.graphics_sample->gather = controls.gather;
      state.graphics_sample->horizontal = controls.horizontal;
      if (controls.compare) {
        state.graphics_sample->comparison = std::make_unique<GraphicsSampleAccess::Comparison>();
        state.graphics_sample->comparison->function = decoded_sampler.compare_function;
      }
    }
  }
  // The functional model retains uncompressed backing for image operations.
  // The memory pipeline materializes GFX11 metadata clears before each access.
  state.buffer_components = atomic ? 0 : gather ? 4 : std::popcount(mask);
  state.buffer_d16 = d16;
  state.buffer_format = format;
  state.buffer_format_encoding = BufferFormatEncoding::Gfx11;
  // Physical GFX11/12 stores select logical channels even with D16. GFX11
  // zero-fills omitted channels; GFX12 replicates the first supplied component.
  const uint32_t channel_mask = mask;
  uint32_t selectors = packed ? 0xfacu : resource_words[3];
  if (state.is_load && !packed)
    selectors = image_load_selectors(selectors, state.decoded_buffer_format);
  uint32_t component = 0;
  for (uint32_t i = 0; i < 4; ++i)
    if (channel_mask & (1u << i))
      state.buffer_selectors |= ((selectors >> (3 * i)) & 7) << (3 * component++);
  if (gather)
    state.buffer_selectors *=
        0b001001001001u; // Repeat the selected channel for four returned texels.
  if (!state.is_load) {
    state.image_store_component_mask = mask;
    state.buffer_selectors = selectors;
  }
  state.wf_size = wf.wf_size();
  state.exec_mask = wf.exec();
  state.elem_size = bytes;
  state.num_elems = 1;
  state.dst_reg_base = wf.vgpr_alloc().base + data;
  const uint32_t body_components = spatial_components + layer_coordinate +
                                   (sample_mode == ImageSampleMode::Explicit || controls.lod_clamp);
  const uint32_t load_components = multisample
                                       ? (type == kImage2dMsaaArray ? 4 : 3)
                                       : spatial_components + layer_coordinate + mip_transfer;
  if (quad_fragments) {
    state.graphics_sample = std::make_unique<GraphicsSampleAccess>();
    state.graphics_sample->tap_count = state.graphics_sample->taps_per_filter = 4;
    state.graphics_sample->gather = state.graphics_sample->horizontal = true;
  }
  const uint32_t coordinate_count =
      !sample ? (a16 ? (load_components + 1) / 2 : load_components)
              : coordinate_offset + (a16 ? (body_components + 1) / 2 : body_components);
  if (!resolve_image_coordinates(wf, coords, coordinate_count))
    return unsupported();
  const uint64_t base = addr_calc::buffer_virtual_address(
                            ((uint64_t{resource_words[1] & 255} << 32) | resource_words[0]) << 8) +
                        mip->offset;
  const uint32_t pitch_field = resource_words[4] & (gfx12 ? 0xffff : 0x3fff);
  // Word 4 describes the last accessible layer for arrays, not custom pitch.
  const uint32_t pitch =
      type == kImage2d && swizzle == 0 && pitch_field ? pitch_field + 1 : mip->pitch;
  const uint64_t resource_base = base - mip->offset;
  const ImageAddressConfig address_config = wf.cu().image_address_config();
  std::unique_ptr<ImageMipAddressCache> device_mips;
  // Ordinary and feedback transfers use the same address equations and base check for
  // uncompressed 2D surfaces.
  const bool shared_2d = (type == kImage2d || type == kImage2dArray) && !compressed && !bc;
  if (!query && swizzle &&
      (shared_2d ||
       (!gfx12 && ((volume && !compressed) || address_config.gb_addr_config != 0x545)))) {
    // The captured DCC/HTILE and multisample equations below describe 0x545 only.
    if (compressed || multisample)
      return unsupported();
    const ImageLayout layout{.width = bc ? (resource_width + 3) / 4 : resource_width,
                             .height = bc ? (resource_height + 3) / 4 : resource_height,
                             .depth = is_array || volume ? descriptor.depth : 1,
                             .element_bytes = bytes,
                             .swizzle_mode = swizzle,
                             .dimensions = volume                           ? 3u
                                           : one_dimensional && !implicit_y ? 1u
                                                                            : 2u,
                             .mip_levels = max_level + 1};
    device_mips = std::make_unique<ImageMipAddressCache>(address_config, layout, resource_base);
    if (!device_mips->get(first_level)) {
      if (!volume && !shared_2d)
        return unsupported();
      wf.report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
      return util::Result::failure();
    }
  }
  const auto texel_address = [&](const ImageMipLayout &layout, uint32_t level, uint32_t layer,
                                 uint32_t x, uint32_t y,
                                 uint32_t row_pitch) -> std::optional<uint64_t> {
    if (bc) {
      x /= 4;
      y /= 4;
    }
    if (device_mips) {
      const ImageAddressing *addressing = device_mips->get(level);
      if (!addressing || !addressing->contains(x, y, layer))
        return std::nullopt;
      return resource_base + addressing->byte_offset(x, y, layer);
    }
    const uint64_t layer_base = image_layer_base(gfx12, resource_base + layout.offset,
                                                 layout.slice_size, layer, bytes, swizzle);
    x += layout.tail_x;
    y += layout.tail_y;
    return gfx12 ? gfx12_image_address(layer_base, x, y, row_pitch, bytes, swizzle)
                 : gfx11_image_address(layer_base, x, y, row_pitch, bytes, swizzle);
  };
  if (compressed && !query) {
    const bool depth = swizzle == 24 || swizzle == 28;
    if ((type != kImage2d && type != kImage3d && type != kImage2dArray) ||
        (depth && (type != kImage2d || max_level)) ||
        (depth ? (bytes != 1 && bytes != 2 && bytes != 4) : (swizzle != 27 && swizzle != 31)))
      return unsupported();
    state.image_metadata = std::make_unique<ImageMetadataAccess>();
    auto &image = *state.image_metadata;
    image.base = base - mip->offset;
    image.slice_size = mip->slice_size;
    image.metadata = addr_calc::buffer_virtual_address((uint64_t{resource_words[7]} << 16) |
                                                       (uint64_t{resource_words[6] >> 24} << 8));
    image.width = resource_width;
    image.height = resource_height;
    image.mip_levels = max_level + 1;
    image.swizzle = swizzle;
    image.pipe_aligned = resource_words[6] & (1u << 19);
    image.depth = depth;
    if (depth && !image.pipe_aligned)
      return unsupported();
  }
  RegisterAccess regs(wf);
  const auto read_coordinate = [&](uint32_t first, uint32_t component, bool packed, uint32_t lane) {
    const uint32_t index = first + (packed ? component / 2 : component);
    const uint32_t half = component % 2;
    const uint8_t bytes = packed ? 3u << (2 * half) : 0xfu;
    const uint32_t value = regs.read_vgpr(wf.vgpr_alloc().base + coords[index], lane, bytes);
    return packed ? (value >> (16 * half)) & 0xffffu : value;
  };
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    const auto load_coordinate = [&](uint32_t component) {
      return read_coordinate(0, component, a16, lane);
    };
    uint32_t x = sample ? 0 : load_coordinate(0);
    uint32_t y = sample || dim == kImageDim1d || dim == kImageDim1dArray ? 0 : load_coordinate(1);
    if (multisample) {
      const uint32_t relative_layer = type == kImage2dMsaaArray ? load_coordinate(2) : 0;
      const uint32_t fragment = load_coordinate(load_components - 1);
      const uint32_t samples = 1u << fragment_log2;
      if (x >= width || y >= height || relative_layer > last_layer - first_layer ||
          (!quad_fragments && fragment >= samples))
        continue;
      const uint32_t surface_pitch =
          type == kImage2dMsaa && pitch_field ? std::max(width, pitch_field + 1) : width;
      const uint32_t layer = first_layer + relative_layer;
      // On W7900 (gfx1100), RGBA8_UINT IMAGE_MSAA_LOAD selects fragments
      // 4..7 for selectors 8 and UINT_MAX on 8x, and repeats 0,1,0,1 on a 2x surface.
      const uint32_t first = quad_fragments ? std::min(fragment, samples - 1) & ~3u : fragment;
      const uint32_t count = quad_fragments ? 4 : 1;
      for (uint32_t i = 0; i < count; ++i) {
        const uint64_t address =
            gfx11_msaa_address(base, x, y, layer, (first + i) % samples, surface_pitch, height,
                               bytes, fragment_log2, swizzle);
        if (quad_fragments) {
          auto &tap = state.graphics_sample->taps[i];
          tap.addresses[lane] = address;
          tap.lane_mask |= uint64_t{1} << lane;
        } else {
          state.per_lane_addr[lane] = address;
        }
      }
      state.lane_mask |= uint64_t{1} << lane;
      continue;
    }
    if (sample) {
      int32_t offset_x = 0, offset_y = 0;
      if (controls.offset && dim != kImageDimCube) {
        const uint32_t offset = read_coordinate(0, 0, false, lane);
        offset_x = int32_t((offset & 63) ^ 32) - 32;
        offset_y = one_dimensional ? 0 : int32_t(((offset >> 8) & 63) ^ 32) - 32;
      }
      if (controls.compare)
        state.graphics_sample->comparison->references[lane] =
            std::bit_cast<float>(read_coordinate(0, compare_offset, false, lane));
      const auto read = [&](uint32_t index, uint32_t source_lane) {
        if (a16 && index >= coordinate_offset) {
          const uint32_t component = index - coordinate_offset;
          return util::f16_to_f32(static_cast<uint16_t>(
              read_coordinate(coordinate_offset, component, true, source_lane)));
        }
        return std::bit_cast<float>(read_coordinate(0, index, false, source_lane));
      };
      double u = read(coordinate_offset, lane),
             v = one_dimensional ? (implicit_y ? 0.0 : 0.5) : read(coordinate_offset + 1, lane);
      uint32_t layer = first_layer, face = 0, cube_base = first_layer;
      const auto cube_token = [&](double selected) -> std::optional<int32_t> {
        if (!std::isfinite(selected) || selected < INT32_MIN || selected > INT32_MAX)
          return std::nullopt;
        return static_cast<int32_t>(selected);
      };
      if (dim == kImageDimCube) {
        const auto token = cube_token(read(coordinate_offset + 2, lane));
        if (!token)
          return unsupported();
        face = std::min(uint32_t(*token) & 7u, 5u);
        const uint32_t cube =
            std::clamp(*token >> 3, 0, int32_t((last_layer - first_layer - 5) / 6));
        cube_base += cube * 6;
        layer = cube_base + face;
        u -= 1;
        v -= 1;
      }
      if ((dim == kImageDim1dArray || dim == kImageDim2dArray) && !query) {
        const double slice = read(coordinate_offset + spatial_components, lane);
        if (!std::isfinite(slice))
          return unsupported();
        // Array slices use nearest-even conversion before view clamping.
        const double lower = std::floor(slice), fraction = slice - lower;
        const double rounded =
            lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0));
        layer += static_cast<uint32_t>(std::clamp(rounded, 0.0, double(last_layer - first_layer)));
      }
      double lod = 0;
      bool zero_footprint = false;
      uint32_t filter_count = 1;
      double sample_step_u = 0, sample_step_v = 0;
      if (sample_mode == ImageSampleMode::Explicit) {
        const double input = read(coordinate_offset + spatial_components + layer_coordinate, lane);
        lod = image_sample_round_even8(std::isnan(input) ? 0 : input) + lod_bias;
      } else if (sample_mode == ImageSampleMode::Zero) {
        lod = lod_bias;
      } else if (needs_gradients) {
        double dxu, dxv, dyu, dyv;
        std::array<int, 4> gradient_rounding{};
        std::array<bool, 4> reflected_coordinates{};
        bool unbounded_cube_footprint = false;
        if (derivatives) {
          // G16 stores (dxu, dxv), then (dyu, dyv), low half first.
          // A16 independently controls the coordinate body after the gradients.
          const auto gradient = [&](uint32_t component) {
            const uint32_t value = read_coordinate(gradient_offset, component, g16, lane);
            return g16 ? util::f16_to_f32(static_cast<uint16_t>(value))
                       : std::bit_cast<float>(value);
          };
          dxu = gradient(0);
          dxv = one_dimensional ? 0 : gradient(1);
          dyu = gradient(one_dimensional && !g16 ? 1 : 2);
          dyv = one_dimensional ? 0 : gradient(3);
        } else {
          const ImageDerivativeQuad quad(lane);
          uint32_t derivative_face = face;
          if (dim == kImageDimCube) {
            const auto selected = cube_token(read(coordinate_offset + 2, quad.origin));
            if (!selected)
              return unsupported();
            derivative_face = std::min(uint32_t(*selected) & 7u, 5u);
          }
          // Unfold onto the quad origin's face. Choosing each lane's face
          // introduces different footprints where three cube faces meet.
          const auto unfolded = [&](uint32_t source_lane) -> std::optional<std::array<double, 2>> {
            const double su = read(coordinate_offset, source_lane);
            const double sv = one_dimensional ? (implicit_y ? 0.0 : 0.5)
                                              : read(coordinate_offset + 1, source_lane);
            const auto saturate_infinity = [](double value) {
              return std::isinf(value)
                         ? std::copysign(double(std::numeric_limits<float>::max()), value)
                         : value;
            };
            if (dim != kImageDimCube)
              return std::array{saturate_infinity(su), saturate_infinity(sv)};
            if (!std::isfinite(su) || !std::isfinite(sv))
              return std::nullopt;
            const auto sf = cube_token(read(coordinate_offset + 2, source_lane));
            if (!sf)
              return std::nullopt;
            const uint32_t source_face = std::min(uint32_t(*sf) & 7u, 5u);
            if (source_face == derivative_face)
              return std::array{su - 1, sv - 1};
            if (source_face / 2 == derivative_face / 2) {
              // Opposite faces select the coarsest available mip.
              unbounded_cube_footprint = true;
              return std::array{u, v};
            }
            // Unfold the adjacent face about its shared edge. Perspective
            // projection would distort the footprint and become singular at
            // perpendicular directions. Physical GFX11/12 derivatives instead
            // retain the distance to the edge in each face's coordinates.
            auto direction = image_cube_direction(source_face, su - 1, sv - 1);
            const double sign = derivative_face & 1 ? -1 : 1;
            direction[source_face / 2] *= 2 - sign * direction[derivative_face / 2];
            direction[derivative_face / 2] = sign;
            const auto policy = image_cube_derivative_policy(derivative_face, source_face);
            // The right neighbor supplies dx; the bottom neighbor supplies dy.
            const uint32_t gradient = source_lane == quad.right() ? 0 : 2;
            for (uint32_t i = 0; i < 2; ++i) {
              gradient_rounding[gradient + i] = policy.rounding_directions[i];
              reflected_coordinates[gradient + i] = policy.reflected_coordinates[i];
            }
            return image_cube_project(derivative_face, direction);
          };
          // Implicit sampling uses the top and left edges of the whole quad,
          // even when this lane is on its bottom or right edge.
          const auto tl = unfolded(quad.origin), tr = unfolded(quad.right()),
                     bl = unfolded(quad.bottom());
          if (!tl || !tr || !bl)
            return unsupported();
          const auto difference = [&](double neighbor, double origin_value, uint32_t component) {
            const double value = neighbor - origin_value;
            return reflected_coordinates[component] ? image_cube_reflected_derivative(value)
                                                    : value;
          };
          dxu = difference((*tr)[0], (*tl)[0], 0);
          dxv = difference((*tr)[1], (*tl)[1], 1);
          dyu = difference((*bl)[0], (*tl)[0], 2);
          dyv = difference((*bl)[1], (*tl)[1], 3);
        }
        // GFX11/12 discard NaN derivatives and saturate infinite gradients.
        // Preserve NaNs until after quad subtraction: replacing a coordinate
        // with zero first would produce a spurious footprint.
        const auto finite_gradient = [](double value) {
          return std::isnan(value) ? 0.0
                 : std::isinf(value)
                     ? std::copysign(double(std::numeric_limits<float>::max()), value)
                     : value;
        };
        dxu = finite_gradient(dxu);
        dxv = finite_gradient(dxv);
        dyu = finite_gradient(dyu);
        dyv = finite_gradient(dyv);
        const auto footprint_axes = image_footprint(dxu, dxv, dyu, dyv, normalized ? width : 1,
                                                    normalized ? height : 1, gradient_rounding);
        const auto [major_lod, minor_lod] = footprint_axes.lods;
        zero_footprint = footprint_axes.norms[0] == 0 && footprint_axes.norms[1] == 0;
        if ((min_filter & 2) || (mag_filter & 2)) {
          lod = std::max(minor_lod, major_lod - std::countr_zero(max_anisotropy) * 256) / 256.0;
          filter_count = image_anisotropic_filter_count(footprint_axes, max_anisotropy,
                                                        aniso_threshold, aniso_bias, perf_mod);
          if (filter_count > 1 && !query) {
            const auto [direction_u, direction_v] =
                footprint_axes.direction(normalized ? width : 1, normalized ? height : 1);
            const auto unbiased_count = image_anisotropic_filter_count(
                footprint_axes, max_anisotropy, aniso_threshold, 0, perf_mod);
            const double spacing = image_anisotropic_filter_step(
                footprint_axes, max_anisotropy, filter_count, unbiased_count, mip_filter == 2);
            sample_step_u = image_filter_step_truncate(direction_u * spacing);
            sample_step_v = image_filter_step_truncate(direction_v * spacing);
          }
        } else {
          lod = major_lod / 256.0;
        }
        lod += lod_bias;
        if (sample_mode == ImageSampleMode::Bias) {
          const uint16_t low16 =
              image_prefix_low16_words(gradient_offset, 0, a16, true, controls.offset);
          const auto bits = regs.read_vgpr(wf.vgpr_alloc().base + coords[bias_offset], lane,
                                           image_prefix_word_bytes(bias_offset, low16));
          const double bias =
              a16 ? util::f16_to_f32(static_cast<uint16_t>(bits)) : std::bit_cast<float>(bits);
          lod += image_sample_round_even8(bias);
        }
        if (unbounded_cube_footprint) {
          lod = max_lod;
          filter_count = 1;
        }
      }
      if (std::isnan(lod))
        lod = 0;
      if (controls.lod_clamp) {
        const double minimum =
            read(coordinate_offset + spatial_components + layer_coordinate, lane);
        const auto clamped = image_shader_min_lod(lod, minimum);
        if (!clamped)
          return unsupported();
        lod = *clamped;
      }
      const double raw_lod = zero_footprint ? 0 : lod;
      lod = std::clamp(lod, min_lod, max_lod);
      const double requested_lod = first_level + lod;
      // LZ gather stays at its fixed mip. Other samples clamp to the resource's
      // resident minimum, independently of the sampler's LOD interval.
      if (controls.gather && sample_mode == ImageSampleMode::Zero)
        lod = 0;
      else
        lod = std::max(lod, resident_min_lod - first_level);
      lod = image_sample_round_even8(lod);
      const uint32_t selected_filter = lod <= 0 ? mag_filter : min_filter;
      const bool linear = selected_filter & 1;
      // TRUNC_COORD selects truncation for point filtering. Otherwise the
      // texture unit rounds to eight fractional bits before selecting a texel.
      const auto texel_origin = [&](double position) {
        return std::floor(!linear && !truncate_coordinates ? image_sample_round_even8(position)
                                                           : position);
      };
      if (!(selected_filter & 2))
        filter_count = 1;
      // Without mip filtering, start at the first resident whole level. Fixed
      // LZ gathers stay at the view base and drop below-residency results.
      if (!mip_filter)
        lod = controls.gather && sample_mode == ImageSampleMode::Zero
                  ? 0.0
                  : std::floor(resident_min_lod) - first_level;
      lod = std::clamp(lod, 0.0, double(last_level - first_level));
      const uint32_t level = first_level + uint32_t(std::floor(lod + (mip_filter == 1 ? 0.5 : 0)));
      const uint32_t mip_fraction =
          mip_filter == 2 ? image_mip_fraction(static_cast<uint32_t>((lod - std::floor(lod)) * 256),
                                               perf_mip, perf_mod)
                          : 0;
      if (query) {
        const double clamped_lod = level - first_level + mip_fraction / 256.0;
        (*queried_lods)[lane] = {static_cast<float>(clamped_lod), static_cast<float>(raw_lod)};
        continue;
      }
      if (controls.gather && level < uint32_t(resident_min_lod))
        continue;
      // Compare the sampler-clamped request before the view and residency clamps.
      // The status carries the level of the first selected mip tap.
      if (state.image_status)
        state.image_status->record_sample(lane, mip_filter != 0, requested_lod,
                                          std::min(last_level, level + (mip_fraction == 256)));
      // Hardware converts NaNs to zero and saturates infinities before
      // address wrapping. Reduce large repeating coordinates before the
      // half-texel offset, which would otherwise disappear in host arithmetic.
      const auto address_input = [&](double value, uint32_t extent, uint32_t wrap) {
        if (std::isnan(value))
          value = 0;
        else if (std::isinf(value))
          value = std::copysign(double(std::numeric_limits<float>::max()), value);
        if (wrap <= 1 && std::abs(value) >= 0x1p32)
          value = std::fmod(value, double(normalized ? 1 : extent) * (wrap == 1 ? 2 : 1));
        return value;
      };
      u = address_input(u, width, wrap_x);
      v = address_input(v, height, wrap_y);
      // Linear filtering reflects the footprint center. Point filtering reflects the
      // quantized texel index, so an exact negative texel boundary selects -1-index.
      if (linear && wrap_x == uint32_t(ImageClamp::MirrorOnce))
        u = std::abs(u);
      if (linear && wrap_y == uint32_t(ImageClamp::MirrorOnce))
        v = std::abs(v);
      const auto address_coordinate = [linear](double value, uint32_t size, uint32_t wrap) {
        const auto mode = static_cast<ImageClamp>(wrap);
        return image_sample_coordinate(
            value, size, linear && mode == ImageClamp::MirrorOnce ? ImageClamp::Edge : mode);
      };
      const auto &lane_mip = get_mip_layout(level);
      if (!lane_mip)
        return unsupported();
      if (state.image_metadata)
        state.image_metadata->layers[lane] = layer;
      if (state.graphics_sample) {
        auto &access = *state.graphics_sample;
        const auto fraction = [](double value) {
          return static_cast<float>(image_sample_round_even8(value));
        };
        access.mip_fractions[lane] = mip_fraction / 256.0f;
        access.filter_counts[lane] = filter_count;
        access.filters.resize(std::max(access.filters.size(), size_t(filter_count)));
        access.tap_count = std::max(access.tap_count, filter_count * access.taps_per_filter);
        if (access.tap_count > GraphicsSampleAccess::kMaxTaps)
          return unsupported();
        access.taps.resize(access.tap_count);
        if (state.image_metadata && max_level)
          state.image_metadata->tap_levels.resize(access.tap_count);
        double sample_u = 0, sample_v = 0, x0 = 0, y0 = 0;
        for (uint32_t tap = 0; tap < filter_count * access.taps_per_filter; ++tap) {
          const uint32_t filter_index = tap / access.taps_per_filter;
          const uint32_t filter_tap = tap % access.taps_per_filter;
          const uint32_t source = filter_tap % access.texels_per_tap;
          const uint32_t texel = filter_tap / access.texels_per_tap;
          const uint32_t mip_index = texel / 4;
          // All taps in a footprint share its center. Each mip also shares
          // the clamped texel origin and weights, including cube helper texels.
          if (filter_tap == 0) {
            const double offset = double(filter_index) - 0.5 * (filter_count - 1);
            sample_u = image_sample_coordinate(u, offset * sample_step_u);
            sample_v = image_sample_coordinate(v, offset * sample_step_v);
          }
          // At either fixed-point endpoint only one mip contributes memory accesses or feedback.
          if (mip_fraction == (mip_index ? 0u : 256u))
            continue;
          const auto &selected = get_mip_layout(std::min(level + mip_index, last_level));
          if (!selected)
            return unsupported();
          if (texel % 4 == 0 && source == 0) {
            double px = sample_u * (normalized ? selected->width : 1) + offset_x -
                        (controls.horizontal ? 1.5
                         : linear            ? 0.5
                                             : 0);
            double py = sample_v * (normalized ? selected->height : 1) + offset_y -
                        (linear && !controls.horizontal ? 0.5 : 0);
            // Clamp to texel centers before generating weights. This preserves
            // exact edge texels, including signed zero in a 1x1 mip level.
            if (linear && !controls.gather && !seamless_cube && (wrap_x == 2 || wrap_x == 3))
              px = std::clamp(px, 0.0, double(selected->width - 1));
            if (linear && !controls.gather && !seamless_cube && (wrap_y == 2 || wrap_y == 3))
              py = std::clamp(py, 0.0, double(selected->height - 1));
            x0 = texel_origin(px);
            y0 = texel_origin(py);
            access.filters[filter_index].fractions[lane][mip_index] =
                linear ? std::array{fraction(px - x0), fraction(py - y0)} : std::array{0.0f, 0.0f};
          }
          if (linear && !controls.gather && !controls.horizontal) {
            const auto &weight = access.filters[filter_index].fractions[lane][mip_index];
            if (weight[0] == ((texel & 1) ? 0.0f : 1.0f) ||
                weight[1] == ((texel & 2) ? 0.0f : 1.0f))
              continue;
          }
          const double raw_x = x0 + (controls.horizontal ? texel : linear ? texel & 1 : 0);
          const double raw_y = y0 + (linear && !controls.horizontal ? (texel >> 1) & 1 : 0);
          std::optional<uint32_t> tx, ty;
          uint32_t selected_layer = layer;
          if (seamless_cube) {
            const bool corner =
                (raw_x < 0 || raw_x >= selected->width) && (raw_y < 0 || raw_y >= selected->height);
            const double cx = std::clamp(raw_x, 0.0, double(selected->width - 1));
            const double cy = std::clamp(raw_y, 0.0, double(selected->height - 1));
            if (corner) {
              access.filters[filter_index].cube_corners[lane][mip_index] |= 1u << (texel % 4);
              // The missing corner texel is shared by the three incident faces.
              if (source == 0) {
                tx = static_cast<uint32_t>(cx);
                ty = static_cast<uint32_t>(cy);
              } else {
                const auto mapped = image_cube_texel(face, source == 1 ? raw_x : cx,
                                                     source == 1 ? cy : raw_y, selected->width);
                tx = mapped.x;
                ty = mapped.y;
                selected_layer = cube_base + mapped.face;
              }
            } else {
              if (source)
                continue;
              const auto mapped = image_cube_texel(face, raw_x, raw_y, selected->width);
              tx = mapped.x;
              ty = mapped.y;
              selected_layer = cube_base + mapped.face;
            }
          } else {
            tx = address_coordinate(raw_x, selected->width, wrap_x);
            ty = address_coordinate(raw_y, selected->height, wrap_y);
          }
          if (!tx || !ty)
            continue;
          const auto address =
              texel_address(*selected, std::min(level + mip_index, last_level), selected_layer, *tx,
                            *ty, max_level ? selected->pitch : pitch);
          if (!address)
            return unsupported();
          access.taps[tap].addresses[lane] = *address;
          access.taps[tap].coordinates[lane] = *tx | (*ty << 16);
          access.taps[tap].layers[lane] = selected_layer;
          if (state.image_metadata && max_level)
            state.image_metadata->tap_levels[tap][lane] = std::min(level + mip_index, last_level);
          access.taps[tap].lane_mask |= uint64_t{1} << lane;
        }
        const uint32_t first_tap = mip_fraction == 256 ? 4 * access.texels_per_tap : 0;
        state.per_lane_addr[lane] = access.taps[first_tap].addresses[lane];
        state.lane_mask |= uint64_t{1} << lane;
        continue;
      }
      x = *address_coordinate(texel_origin(u * (normalized ? lane_mip->width : 1) + offset_x),
                              lane_mip->width, wrap_x);
      y = *address_coordinate(texel_origin(v * (normalized ? lane_mip->height : 1) + offset_y),
                              lane_mip->height, wrap_y);
      const auto address =
          texel_address(*lane_mip, level, layer, x, y, max_level ? lane_mip->pitch : pitch);
      if (!address)
        return unsupported();
      state.per_lane_addr[lane] = *address;
      if (bc)
        state.image_bc_texels[lane] = (x & 3) + ((y & 3) << 2);
      if (state.image_metadata) {
        state.image_metadata->coordinates[lane] = x | (y << 16);
        state.image_metadata->layers[lane] = layer;
        state.image_metadata->levels[lane] = level;
      }
      state.lane_mask |= uint64_t{1} << lane;
      continue;
    }
    const uint32_t relative_layer = dim == kImageDim2dArray || dim == kImageDimCube
                                        ? load_coordinate(2)
                                    : dim == kImageDim1dArray ? load_coordinate(1)
                                                              : 0;
    if (!is_array && !volume && relative_layer)
      return unsupported();
    // Explicit mip transfers supply an unsigned, view-relative LOD per lane.
    // Reject it before adding the base level, including values that would wrap.
    const uint32_t relative_level = mip_transfer ? load_coordinate(load_components - 1) : 0;
    if (relative_level > last_level - first_level)
      continue;
    const uint32_t level = first_level + relative_level;
    if (state.image_status)
      state.image_status->levels[lane] = level;
    if (volume) {
      const auto selected =
          image_volume_mip_layout(gfx12, swizzle, bytes, resource_width, resource_height,
                                  resource_depth, max_level + 1, level);
      if (!selected)
        return unsupported();
      const uint32_t z = load_coordinate(2);
      if (device_mips && !device_mips->get(level))
        return unsupported();
      const auto address =
          device_mips
              ? texel_address(selected->plane, level, z, x, y, selected->plane.pitch)
              : image_volume_address(gfx12, resource_base, *selected, x, y, z, bytes, swizzle);
      if (address) {
        state.per_lane_addr[lane] = *address;
        if (state.image_metadata) {
          // Render-optimized 3D tiles use the same DCC layout as array slices.
          state.image_metadata->coordinates[lane] = x | (y << 16);
          state.image_metadata->layers[lane] = load_coordinate(2);
          state.image_metadata->levels[lane] = level;
        }
        state.lane_mask |= uint64_t{1} << lane;
      }
      continue;
    }
    const auto &selected = get_mip_layout(level);
    if (!selected)
      return unsupported();
    if (x >= selected->width || y >= selected->height ||
        (is_array && relative_layer > last_layer - first_layer))
      continue;
    const uint32_t layer = is_array ? first_layer + relative_layer : 0;
    const auto address =
        texel_address(*selected, level, layer, x, y, max_level ? selected->pitch : pitch);
    if (!address)
      return unsupported();
    state.per_lane_addr[lane] = *address;
    if (bc)
      state.image_bc_texels[lane] = (x & 3) + ((y & 3) << 2);
    if (state.image_metadata) {
      state.image_metadata->coordinates[lane] = x | (y << 16);
      state.image_metadata->layers[lane] = layer;
      state.image_metadata->levels[lane] = level;
    }
    state.lane_mask |= uint64_t{1} << lane;
  }
  if (dropped) {
    state.lane_mask = 0;
    state.buffer_components = 1;
    state.buffer_selectors = 0;
    state.image_sampling = false;
    state.graphics_sample.reset();
    state.image_metadata.reset();
  }
  if (atomic) {
    // Image atomic payloads are raw words. DMASK includes the comparison
    // operand, while the return value occupies only one memory element.
    const unsigned atomic_words = std::popcount(mask);
    state.store_data.resize(wf.wf_size() * atomic_words * 4);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (uint64_t{1} << lane)))
        continue;
      for (uint32_t word = 0; word < atomic_words; ++word) {
        const uint32_t source = gfx115_source_vgpr(data + word, wf.num_vgprs());
        const uint32_t value = regs.read_vgpr(wf.vgpr_alloc().base + source, lane);
        std::memcpy(state.store_data.data() + (lane * atomic_words + word) * 4, &value, 4);
      }
    }
  } else if (!state.is_load)
    capture_buffer_format_store(wf, state, state.dst_reg_base, packed);
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu
