// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/result.h"

#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Encoded image access controls and coordinate VGPR selectors.
struct ImageFields {
  uint32_t dmask = 0;
  uint32_t dim = 0;
  bool a16 = false;
  bool d16 = false;
  bool r128 = false;
  bool tfe = false;
  bool lwe = false;
  bool da = false;
  bool unorm = false;
  bool mip = false;    ///< The address tuple includes a view-relative mip level.
  bool gather = false; ///< Return four selected texel components.
};

/// @brief Decode and qualify the descriptor and return its formatted-transfer state.
inline util::FailureOr<ImageResource> prepare_image_resource(Wavefront &wf, VectorMemState &state,
                                                             uint32_t resource_reg,
                                                             ImageFields fields) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  const uint64_t exec = wf.exec();
  state.wf_size = wf.wf_size();
  state.wg_id = wf.wg_id();
  state.wf_id = wf.wf_id();
  state.exec_mask = exec;
  state.lane_mask = 0;
  state.per_lane_addr.fill(0);
  state.num_elems = 1;
  state.elem_size = 1;
  state.buffer_d16 = fields.d16;
  state.buffer_components = static_cast<uint32_t>(std::popcount(fields.dmask & 15u));
  const rj_code_arch_t arch = wf.cu().arch();
  if (arch != ROCJITSU_CODE_ARCH_CDNA1 && arch != ROCJITSU_CODE_ARCH_CDNA2 &&
      arch != ROCJITSU_CODE_ARCH_RDNA1 && arch != ROCJITSU_CODE_ARCH_RDNA2 &&
      arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  state.buffer_format_encoding = gfx9                               ? BufferFormatEncoding::Gfx9
                                 : arch == ROCJITSU_CODE_ARCH_RDNA1 ? BufferFormatEncoding::Rdna1
                                 : arch == ROCJITSU_CODE_ARCH_RDNA2 ? BufferFormatEncoding::Rdna2
                                                                    : BufferFormatEncoding::Gfx11;
  if (fields.r128 || fields.tfe || fields.lwe)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.buffer_components == 0 || (!state.is_load && !fields.unorm))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (!scalar_selector_range_is_backed(wf, resource_reg, 8))
    return ImageResource{};

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = read_scalar_selector(wf, resource_reg + i);
  const ImageResource resource = decode_image_resource(words, arch);

  if (resource.type < kImage1d || resource.data_format == 0)
    return ImageResource{};

  if (resource.compressed || resource.base_level > resource.last_level ||
      resource.last_level > resource.max_level || resource.max_level >= 17)
    return fail(InstructionExecutionError::UnimplementedInstruction);

  // DMASK packs supplied shader components into consecutive registers (or
  // halves for D16). Loads compact their selectors to produce that tuple.
  // RDNA stores retain the descriptor's shader-component slots and scatter
  // their packed inputs back into those slots before applying the full
  // mapping. CDNA2 retains its compacted-selector store behavior.
  const uint32_t descriptor_selectors = resource.selectors;
  if (state.is_load || gfx9) {
    uint32_t selectors = 0;
    for (uint32_t component = 0, packed = 0; component < 4; ++component)
      if (fields.dmask & (1u << component))
        selectors |= ((descriptor_selectors >> (3 * component)) & 7) << (3 * packed++);
    state.buffer_selectors = selectors;
  } else {
    state.buffer_selectors = descriptor_selectors;
    state.image_store_component_mask = fields.dmask & 15u;
  }

  // The six-bit image DFMT must fit the buffer decoder's four-bit packed key.
  if (gfx9 && resource.data_format > 15)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  state.buffer_format = resource.data_format | (resource.num_format << 4);
  state.image_srgb = false;
  if (gfx9 && resource.num_format == 9) {
    state.image_srgb = true;
    state.buffer_format = resource.data_format; // Decode channels as UNORM first.
  } else if (!gfx9) {
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    const uint32_t first_srgb = gfx10 ? 128 : 64;
    if (resource.data_format >= first_srgb && resource.data_format <= first_srgb + 2) {
      constexpr std::array<uint32_t, 3> unorm{1, 14, 42};
      state.image_srgb = true;
      state.buffer_format = unorm[resource.data_format - first_srgb];
      if (gfx10 && resource.data_format == first_srgb + 2)
        state.buffer_format += 14;
    }
  }
  if (state.image_srgb && !state.is_load)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const util::FailureOr<bool> format = prepare_buffer_format_state(state);
  // Buffer decoder failure does not imply an illegal image format: images
  // also support formats such as SRGB that the buffer table does not model.
  if (format.failed())
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!format.value())
    return ImageResource{};

  // CDNA2 preserves its legacy compacted-selector acceptance; only RDNA's
  // full-selector/scatter path requires a qualified inverse mapping.
  if (!gfx9 && !state.is_load &&
      !image_store_selectors_supported(state.decoded_buffer_format, descriptor_selectors))
    return fail(InstructionExecutionError::UnimplementedInstruction);

  return resource;
}

/// @brief Prepare IMAGE_LOAD/STORE with optional view-relative mip operands for the vector memory
/// pipeline.
/// @details The descriptor selects format, bounds, row pitch and tiling;
/// coordinates may come from a consecutive tuple or NSA VGPRs.
inline util::Result prepare_image_access(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg, const ImageAddressRegisters &vaddr,
                                         ImageFields fields) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  const auto prepared = prepare_image_resource(wf, state, resource_reg, fields);
  if (prepared.failed())
    return util::Result::failure();
  const auto &resource = prepared.value();
  if (!resource.width)
    return util::Result::success();
  const uint64_t exec = wf.exec();
  const rj_code_arch_t arch = wf.cu().arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool layered = resource.type == kImage1dArray || resource.type == kImage2dArray;
  if (resource.type == kImageCube || resource.type > kImage2dArray || resource.compressed ||
      resource.base_array != 0 || resource.array_pitch != 0)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  // Execution consumes the spatial prefix selected by the resource type,
  // which can be shorter than the DIM-encoded coordinate list. The compatibility
  // checks below require DIM to supply enough coordinates, not an equal count;
  // for example, a 2D resource with 2D_ARRAY DIM uses X/Y and ignores layer.
  const bool has_y = resource.type != kImage1d && resource.type != kImage1dArray;
  const bool has_slice = resource.type == kImage3d || (layered && (fields.da || !gfx9));
  // CLR can address a 1D array with a 2D_ARRAY instruction. DIM fixes the
  // layer operand's position even when the resource ignores the Y coordinate.
  const uint32_t slice_coordinate =
      !gfx9 && resource.type == kImage1dArray &&
              (fields.dim == kImageDim2dArray || fields.dim == kImageDim3d)
          ? 2u
      : has_y ? 2u
              : 1u;
  const uint32_t coordinates = has_slice ? slice_coordinate + 1 : 1 + (has_y ? 1 : 0);
  if (!gfx9) {
    const uint32_t dim = fields.dim & 7u;
    // A plain spatial DIM has no layer operand. Reject it rather than
    // misinterpreting Y as a layer on a 1D-array resource.
    if (layered && dim != kImageDim1dArray && dim != kImageDim2dArray && dim != kImageDim3d)
      return fail(InstructionExecutionError::UnimplementedInstruction);
    uint32_t dim_spatial_coordinates = 0;
    switch (dim) {
    case kImageDim1d:
    case kImageDim1dArray:
      dim_spatial_coordinates = 1;
      break;
    case kImageDim2d:
    case kImageDim2dArray:
      dim_spatial_coordinates = 2;
      break;
    case kImageDim3d:
      dim_spatial_coordinates = 3;
      break;
    default: // CUBE and MSAA dimensions are not implemented.
      return fail(InstructionExecutionError::UnimplementedInstruction);
    }
    // CLR accesses thin 3D copies with a 2D_ARRAY instruction. Both it and
    // 3D DIM supply X/Y plus the slice coordinate consumed by that resource.
    const uint32_t resource_spatial_coordinates = has_y ? 2 : 1;
    const uint32_t dim_coordinates = image_coordinate_count(dim);
    if (dim_spatial_coordinates < resource_spatial_coordinates || dim_coordinates < coordinates)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  }
  const uint32_t mip_coordinate = gfx9 ? coordinates : image_coordinate_count(fields.dim);
  const uint32_t inputs = fields.mip ? mip_coordinate + 1 : coordinates;
  const uint32_t registers = fields.a16 ? (inputs + 1) / 2 : inputs;
  for (uint32_t i = 0; i < registers; ++i)
    if (vaddr[i] >= wf.num_vgprs())
      return fail(InstructionExecutionError::UnsupportedOperandValue);
  RegisterAccess regs(wf);
  const uint32_t depth = layered || resource.type == kImage3d ? resource.depth : 1;
  const ImageLayout layout{.width = resource.width,
                           .height = has_y ? resource.height : 1u,
                           .depth = depth,
                           .pitch = resource.pitch,
                           .element_bytes = state.elem_size,
                           .swizzle_mode = resource.swizzle_mode,
                           .dimensions = resource.type == kImage3d ? 3u
                                         : has_y                   ? 2u
                                                                   : 1u,
                           .mip_levels = resource.max_level + 1};
  std::array<std::optional<ImageAddressing>, 17> addresses;
  addresses[resource.base_level] =
      ImageAddressing::prepare(wf.cu().image_address_config(), layout, resource.base_level);
  if (!addresses[resource.base_level] ||
      !addresses[resource.base_level]->has_supported_base(resource.base))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> coords{};
    for (uint32_t i = 0; i < inputs; ++i)
      coords[i] =
          fields.a16
              ? (regs.read_vgpr(wf.vgpr_alloc().base + vaddr[i / 2], lane) >> (16 * (i % 2))) &
                    0xffffu
              : regs.read_vgpr(wf.vgpr_alloc().base + vaddr[i], lane);
    const uint32_t x = coords[0];
    const uint32_t y = has_y ? coords[1] : 0;
    const uint32_t slice = has_slice ? coords[slice_coordinate] : 0;
    const uint64_t level64 =
        uint64_t{resource.base_level} + (fields.mip ? coords[mip_coordinate] : 0);
    if (level64 > resource.last_level)
      continue;
    const uint32_t level = level64;
    if (level > resource.last_level || x >= std::max(1u, resource.width >> level) ||
        y >= std::max(1u, resource.height >> level) ||
        slice >= (resource.type == kImage3d ? std::max(1u, depth >> level) : depth))
      continue;
    if (!addresses[level])
      addresses[level] = ImageAddressing::prepare(wf.cu().image_address_config(), layout, level);
    if (!addresses[level])
      return fail(InstructionExecutionError::UnimplementedInstruction);
    if (!addresses[level]->contains(x, y, slice))
      continue;
    const uint64_t offset = addresses[level]->byte_offset(x, y, slice);
    state.per_lane_addr[lane] = addr_calc::buffer_virtual_address(resource.base + offset);
    state.lane_mask |= uint64_t{1} << lane;
  }
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu
