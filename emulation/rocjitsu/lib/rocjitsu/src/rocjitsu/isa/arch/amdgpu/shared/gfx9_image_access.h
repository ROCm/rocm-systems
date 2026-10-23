// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
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

/// @brief MIMG fields controlling GFX9 linear image access.
struct Gfx9ImageFields {
  uint32_t dmask = 0;
  bool a16 = false;
  bool d16 = false;
  bool lwe = false;
  bool da = false;
  bool unorm = false;
};

/// @brief Set up a GFX9 IMAGE_LOAD or IMAGE_STORE as a formatted vector access.
/// @details Each active lane addresses one texel of a linear, single-level
/// image with unnormalized coordinates read from @p vaddr. Lanes outside the
/// image are left out of the lane mask, so a load returns zero and a store is
/// dropped. A null resource drops every lane. The texel layout is the one
/// addrlib gives a linear surface: rows of PITCH elements, one row per 1D slice
/// and HEIGHT rows per 2D slice. DMASK picks the shader components after the
/// descriptor's channel selects, packed into consecutive VGPRs; a store
/// inverts those selects the way formatted buffer stores do. D16 consumes packed
/// half registers. DA=0 accesses layer zero without a layer coordinate.
/// @returns Failure after reporting an unsupported operand (empty DMASK,
/// UNORM=0 store) or an unimplemented feature (tiling, mips, cube, MSAA,
/// nonzero base array, format outside the buffer table, or LWE).
inline util::Result prepare_gfx9_image_access(Wavefront &wf, VectorMemState &state,
                                              uint32_t resource_reg, uint32_t vaddr,
                                              Gfx9ImageFields fields) {
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
  state.buffer_format_encoding = BufferFormatEncoding::Gfx9;
  if (fields.lwe)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.buffer_components == 0 || (!state.is_load && !fields.unorm))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (!scalar_selector_range_is_backed(wf, resource_reg, 8))
    return util::Result::success();

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = read_scalar_selector(wf, resource_reg + i);
  const Gfx9ImageResource resource = decode_gfx9_image_resource(words);

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
      resource.swizzle_mode != 0 || resource.base_level != 0 || resource.last_level != 0 ||
      resource.base_array != 0 || resource.array_pitch != 0)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  // The six-bit image DFMT must fit the buffer decoder's four-bit packed key.
  if (resource.data_format > 15)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  state.buffer_format = resource.data_format | (resource.num_format << 4);
  const auto format = prepare_buffer_format_state(state);
  // Buffer decoder failure does not imply an illegal image format: images
  // also support formats such as SRGB that the buffer table does not model.
  if (format.failed())
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!format.value())
    return util::Result::success();

  const bool has_y = resource.type != kGfx9Image1d && resource.type != kGfx9Image1dArray;
  const bool has_slice = resource.type == kGfx9Image3d || (layered && fields.da);
  const uint32_t coordinates = 1 + (has_y ? 1 : 0) + (has_slice ? 1 : 0);
  const uint32_t slice_rows = has_y ? resource.height : 1;
  RegisterAccess regs(wf);
  const auto addresses = regs.read_vgpr_region(
      wf.vgpr_alloc().base + vaddr, fields.a16 ? (coordinates + 1) / 2 : coordinates, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 3> coords{};
    for (uint32_t i = 0; i < coordinates; ++i)
      coords[i] = fields.a16 ? (addresses.lane(i / 2, lane) >> (16 * (i % 2))) & 0xffffu
                             : addresses.lane(i, lane);
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

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_
