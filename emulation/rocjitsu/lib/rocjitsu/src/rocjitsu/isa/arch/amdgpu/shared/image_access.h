// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/result.h"

#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Instruction fields controlling linear image access.
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
};

/// @brief Set up IMAGE_LOAD or IMAGE_STORE as a formatted vector access.
/// @details Each active lane addresses one texel of a linear, single-level
/// GFX9, GFX11 or GFX12 image with unnormalized coordinates read from
/// @p address_registers. Lanes outside the
/// image are left out of the lane mask, so a load returns zero and a store is
/// dropped. A null resource drops every lane. The texel layout is the one
/// addrlib gives a linear surface: rows of PITCH elements, one row per 1D slice
/// and HEIGHT rows per 2D slice. DMASK picks the shader components after the
/// descriptor's channel selects, packed into consecutive VGPRs. Loads and CDNA2
/// stores compact those selectors into the enabled DMASK slots. RDNA stores
/// instead scatter the packed inputs back into their shader-component slots,
/// qualify the descriptor's full selector mapping, and invert it the way
/// formatted buffer stores do. D16 consumes packed half registers. DA=0 accesses
/// layer zero without a layer coordinate.
/// @returns Failure after reporting an unsupported operand (empty DMASK,
/// UNORM=0 store) or an unimplemented feature (R128, an architecture other than
/// CDNA2/RDNA3/RDNA3.5/RDNA4, tiling, mips, cube, MSAA, RDNA linear
/// 1D/2D-array and 3D resources whose decoded descriptors do not provide PITCH,
/// decoded RDNA PITCH narrower than WIDTH, nonzero base array, incompatible
/// RDNA DIM, format outside the buffer table, unqualified RDNA full-selector
/// store mappings, TFE, or LWE).
inline util::Result prepare_image_access(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg,
                                         const ImageAddressRegisters &address_registers,
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
  if (fields.r128)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (fields.tfe || fields.lwe)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.buffer_components == 0 || (!state.is_load && !fields.unorm))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const auto arch = wf.cu().arch();
  const bool gfx9 = arch == ROCJITSU_CODE_ARCH_CDNA2;
  ImageResource (*decode_resource)(const std::array<uint32_t, 8> &) = nullptr;
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
    state.buffer_format_encoding = BufferFormatEncoding::Gfx9;
    decode_resource = decode_gfx9_image_resource;
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    state.buffer_format_encoding = BufferFormatEncoding::Gfx11;
    decode_resource = decode_gfx11_image_resource;
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    state.buffer_format_encoding = BufferFormatEncoding::Gfx11;
    decode_resource = decode_gfx12_image_resource;
    break;
  default:
    return fail(InstructionExecutionError::UnimplementedInstruction);
  }
  if (!scalar_selector_range_is_backed(wf, resource_reg, 8))
    return util::Result::success();

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = read_scalar_selector(wf, resource_reg + i);
  const ImageResource resource = decode_resource(words);

  if (resource.type < kImage1d || resource.data_format == 0)
    return util::Result::success();

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

  const bool layered = resource.type == kImage1dArray || resource.type == kImage2dArray;
  if (resource.type == kImageCube || resource.type > kImage2dArray || resource.swizzle_mode != 0 ||
      resource.base_level != 0 || resource.last_level != 0 || resource.base_array != 0 ||
      resource.array_pitch != 0 || resource.pitch == 0)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!gfx9 && resource.pitch < resource.width)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  // The six-bit image DFMT must fit the buffer decoder's four-bit packed key.
  if (gfx9 && resource.data_format > 15)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  state.buffer_format =
      gfx9 ? resource.data_format | (resource.num_format << 4) : resource.data_format;
  const auto format = prepare_buffer_format_state(state);
  // Buffer decoder failure does not imply an illegal image format: images
  // also support formats such as SRGB that the buffer table does not model.
  if (format.failed())
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!format.value())
    return util::Result::success();
  // CDNA2 preserves its legacy compacted-selector acceptance; only RDNA's
  // full-selector/scatter path requires a qualified inverse mapping.
  if (!gfx9 && !state.is_load &&
      !image_store_selectors_supported(state.decoded_buffer_format, descriptor_selectors))
    return fail(InstructionExecutionError::UnimplementedInstruction);

  const bool has_y = resource.type != kImage1d && resource.type != kImage1dArray;
  const bool has_slice = resource.type == kImage3d || (layered && fields.da);
  const uint32_t coordinates = 1 + (has_y ? 1 : 0) + (has_slice ? 1 : 0);
  if (!gfx9) {
    const uint32_t dim = fields.dim & 7u;
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
    const uint32_t resource_spatial_coordinates = resource.type == kImage3d ? 3 : has_y ? 2 : 1;
    const uint32_t dim_coordinates = image_coordinate_count(dim);
    if (dim_spatial_coordinates < resource_spatial_coordinates || dim_coordinates < coordinates)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  }
  const uint32_t address_words = fields.a16 ? (coordinates + 1) / 2 : coordinates;
  const uint32_t slice_rows = has_y ? resource.height : 1;
  RegisterAccess regs(wf);
  std::array<std::array<uint32_t, 64>, 3> addresses{};
  for (uint32_t word = 0; word < address_words; ++word) {
    const auto values =
        regs.read_vgpr_region(wf.vgpr_alloc().base + address_registers[word], 1, exec);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      if (exec & (uint64_t{1} << lane))
        addresses[word][lane] = values.lane(0, lane);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 3> coords{};
    for (uint32_t i = 0; i < coordinates; ++i)
      coords[i] =
          fields.a16 ? (addresses[i / 2][lane] >> (16 * (i % 2))) & 0xffffu : addresses[i][lane];
    const uint32_t x = coords[0];
    const uint32_t y = has_y ? coords[1] : 0;
    const uint32_t slice = has_slice ? coords[has_y ? 2 : 1] : 0;
    if (x >= resource.width || y >= resource.height || slice >= resource.depth)
      continue;
    const uint64_t element = (uint64_t{slice} * slice_rows + y) * resource.pitch + x;
    state.per_lane_addr[lane] =
        addr_calc::buffer_virtual_address(resource.base + element * state.elem_size);
    state.lane_mask |= uint64_t{1} << lane;
  }
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu
