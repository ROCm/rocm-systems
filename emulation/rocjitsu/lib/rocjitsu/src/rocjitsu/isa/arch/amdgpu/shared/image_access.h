// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ACCESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ACCESS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/result.h"

#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Encoded image access controls and coordinate VGPR selectors.
struct ImageAccessFields {
  uint32_t dmask = 0;
  bool a16 = false;
  bool d16 = false;
  bool lwe = false;
  bool da = false;
  bool unorm = false;
  uint32_t dim = ~0u; ///< GFX10+ DIM encoding; all ones means the GFX9 descriptor supplies it.
  bool unsupported = false; ///< Unsupported encoding controls, including R128/TFE.
};

/// @brief Prepare IMAGE_LOAD/STORE from a selected mip view for the vector memory pipeline.
/// @details The descriptor selects format, allocation mip chain, view level, bounds and tiling;
/// coordinates may come from a consecutive tuple or NSA VGPRs.
inline util::Result prepare_image_access(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg, std::array<uint32_t, 3> vaddr,
                                         ImageAccessFields fields) {
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
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  state.buffer_format_encoding = gfx9                               ? BufferFormatEncoding::Gfx9
                                 : arch == ROCJITSU_CODE_ARCH_RDNA1 ? BufferFormatEncoding::Rdna1
                                 : arch == ROCJITSU_CODE_ARCH_RDNA2 ? BufferFormatEncoding::Rdna2
                                                                    : BufferFormatEncoding::Gfx11;
  if (fields.lwe || fields.unsupported)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.buffer_components == 0 || (!state.is_load && !fields.unorm))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (!scalar_selector_range_is_backed(wf, resource_reg, 8))
    return util::Result::success();

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = read_scalar_selector(wf, resource_reg + i);
  const ImageResource resource = decode_image_resource(words, arch);

  if (resource.type < kGfx9Image1d || resource.data_format == 0)
    return util::Result::success();

  // DMASK selects memory components; D16 packs their values into consecutive
  // half registers. The ambiguous GFX9 word-count wording also appears in
  // RDNA2, whose Table 41 explicitly illustrates sparse R/A selection.
  uint32_t selectors = 0;
  for (uint32_t component = 0, packed = 0; component < 4; ++component)
    if (fields.dmask & (1u << component))
      selectors |= ((resource.selectors >> (3 * component)) & 7) << (3 * packed++);
  state.buffer_selectors = selectors;

  const bool layered = resource.type == kGfx9Image1dArray || resource.type == kGfx9Image2dArray;
  if (resource.type == kGfx9ImageCube || resource.type > kGfx9Image2dArray ||
      resource.base_level != resource.last_level || resource.last_level > resource.max_level ||
      resource.compressed || resource.base_array != 0 || resource.array_pitch != 0)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  // The six-bit image DFMT must fit the buffer decoder's four-bit packed key.
  if (gfx9 && resource.data_format > 15)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  state.buffer_format = resource.data_format | (resource.num_format << 4);
  const util::FailureOr<bool> format = prepare_buffer_format_state(state);
  // Buffer decoder failure does not imply an illegal image format: images
  // also support formats such as SRGB that the buffer table does not model.
  if (format.failed())
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!format.value())
    return util::Result::success();

  // DIM uses the low three bits of the SQ resource-type enumeration.
  const uint32_t dimension_type = fields.dim + kGfx9Image1d;
  const bool has_y = gfx9 ? resource.type != kGfx9Image1d && resource.type != kGfx9Image1dArray
                          : dimension_type != kGfx9Image1d && dimension_type != kGfx9Image1dArray;
  const bool has_slice = gfx9 ? resource.type == kGfx9Image3d || (layered && fields.da)
                              : dimension_type == kGfx9Image3d ||
                                    dimension_type == kGfx9Image1dArray ||
                                    dimension_type == kGfx9Image2dArray;
  if (!gfx9 && (dimension_type == kGfx9ImageCube || dimension_type == kGfx9Image2dMsaa ||
                dimension_type == kGfx9Image2dMsaaArray))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!gfx9 &&
      (fields.dim > 7 || (dimension_type == kGfx9Image3d && resource.type != kGfx9Image3d)))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  const uint32_t coordinates = 1 + has_y + has_slice;
  const uint32_t registers = fields.a16 ? (coordinates + 1) / 2 : coordinates;
  for (uint32_t i = 0; i < registers; ++i)
    if (vaddr[i] >= wf.num_vgprs())
      return fail(InstructionExecutionError::UnsupportedOperandValue);
  RegisterAccess regs(wf);
  const uint32_t depth = layered || resource.type == kGfx9Image3d ? resource.depth : 1;
  const ImageLayout layout{.width = resource.width,
                           .height = has_y ? resource.height : 1u,
                           .depth = depth,
                           .pitch = resource.pitch,
                           .element_bytes = state.elem_size,
                           .swizzle_mode = resource.swizzle_mode,
                           .dimensions = resource.type == kGfx9Image3d ? 3u
                                         : has_y                       ? 2u
                                                                       : 1u,
                           .mip_levels = resource.max_level + 1};
  const ImageAddressConfig address_config = wf.cu().image_address_config();
  const std::optional<ImageAddressing> addressing =
      ImageAddressing::prepare(address_config, layout, resource.base_level);
  if (!addressing)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 3> coords{};
    for (uint32_t i = 0; i < coordinates; ++i)
      coords[i] =
          fields.a16
              ? (regs.read_vgpr(wf.vgpr_alloc().base + vaddr[i / 2], lane) >> (16 * (i % 2))) &
                    0xffffu
              : regs.read_vgpr(wf.vgpr_alloc().base + vaddr[i], lane);
    const uint32_t x = coords[0];
    const uint32_t y = has_y ? coords[1] : 0;
    const uint32_t slice = has_slice ? coords[has_y ? 2 : 1] : 0;
    if (!addressing->contains(x, y, slice))
      continue;
    state.per_lane_addr[lane] =
        addr_calc::buffer_virtual_address(resource.base + addressing->byte_offset(x, y, slice));
    state.lane_mask |= uint64_t{1} << lane;
  }
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ACCESS_H_
