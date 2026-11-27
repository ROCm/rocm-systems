// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/graphics/bvh_intersection.h"

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/data_types.h"

#include <bit>

namespace rocjitsu::amdgpu {

util::Result prepare_gfx11_bvh_intersection(Wavefront &wf, VectorMemState &state,
                                            uint32_t descriptor, uint32_t destination,
                                            uint32_t address, const uint32_t *nsa, bool wide,
                                            bool a16, bool valid_encoding) {
  const auto fail = [&](const char *detail) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue,
                                          detail);
    return util::Result::failure();
  };
  if (!valid_encoding || !scalar_selector_range_is_backed(wf, descriptor, 4)) {
    return fail("invalid image_bvh encoding or resource tuple");
  }
  uint32_t word[4];
  for (unsigned i = 0; i < 4; ++i) {
    word[i] = read_scalar_selector(wf, descriptor + i);
  }
  const auto last_node = gfx11_bvh_last_node(word[2], word[3]);
  if (!last_node) {
    return fail("invalid BVH resource type");
  }
  const uint64_t base = (word[0] | (uint64_t(word[1] & 255) << 32)) << 8;
  auto access = std::make_unique<BvhIntersectionAccess>();
  access->controls.pointer_flags = gfx11_bvh_pointer_flags_enabled(word[3]);
  access->controls.grow = uint8_t(word[1] >> 23);
  access->controls.sort = (word[1] >> 21) & 3;
  access->controls.sort_enabled = word[1] >> 31;
  access->controls.barycentrics = bool(word[3] & (1u << 24));

  const auto registers = gfx11_bvh_registers(wide, a16, address, nsa);
  const unsigned extent = 1u + wide, origin = extent + 1, direction = origin + 3;
  state.wf_size = wf.wf_size();
  state.wg_id = wf.wg_id();
  state.wf_id = wf.wf_id();
  state.dst_reg_base = wf.vgpr_alloc().base + destination;
  state.exec_mask = RegisterAccess(wf).owns_vgpr_range(state.dst_reg_base, 4) ? wf.exec() : 0;
  // A box32 spans two 64-byte nodes; the other defined fetches span one.
  state.elem_size = 64;
  uint64_t box32 = 0;
  RegisterAccess regs(wf);
  for (uint64_t mask = state.exec_mask; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto &ray = access->rays[lane];
    const auto read = [&](unsigned word_index) {
      const unsigned reg = gfx115_source_vgpr(registers[word_index], wf.num_vgprs());
      return regs.read_vgpr(wf.vgpr_alloc().base + reg, lane);
    };
    uint64_t pointer = read(0);
    if (wide) {
      pointer |= uint64_t(read(1)) << 32;
    }
    ray.node_type = uint32_t(gfx11_bvh_node_type(pointer));
    ray.flags = access->controls.pointer_flags ? uint16_t(pointer >> 54) : 0;
    if (!bvh_node_can_load(pointer, *last_node, access->controls.pointer_flags)) {
      continue;
    }
    const uint64_t index = gfx11_bvh_node_index(pointer);
    state.per_lane_addr[lane] = addr_calc::buffer_virtual_address(base + index * 64);
    state.lane_mask |= uint64_t{1} << lane;
    if (ray.node_type == uint32_t(Gfx11BvhNodeType::Box32)) {
      box32 |= uint64_t{1} << lane;
    }
    ray.extent = std::bit_cast<float>(read(extent));
    for (unsigned ax = 0; ax < 3; ++ax) {
      ray.origin[ax] = std::bit_cast<float>(read(origin + ax));
      const auto dir = read(direction + ax);
      ray.direction[ax] = a16 ? util::f16_to_f32(uint16_t(dir)) : std::bit_cast<float>(dir);
      ray.inverse[ax] = a16 ? util::f16_to_f32(uint16_t(dir >> 16))
                            : std::bit_cast<float>(read(direction + 3 + ax));
    }
  }
  state.num_elems = box32 ? 2 : 1;
  if (box32 && box32 != state.lane_mask) {
    state.element_lane_masks.assign(2, state.lane_mask);
    state.element_lane_masks[1] = box32;
  }
  state.bvh = std::move(access);
  return util::Result::success();
}

void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state) {
  if (!cu.owns_vgpr_range(wf, state.dst_reg_base, 4)) {
    return;
  }
  alignas(64) uint32_t result[4][64];
  uint32_t *d[] = {result[0], result[1], result[2], result[3]};
  const auto &access = *state.bvh;
  const uint32_t stride = state.elem_size * state.num_elems;
  const uint64_t flags = wf.cu().arithmetic_flags();
  const int status =
      state.wf_size == 32
          ? rj_arith_image_bvh_intersect_ray_w32(
                flags, uint32_t(state.exec_mask), 0, d, state.response_data.data(), stride,
                uint32_t(state.lane_mask), access.rays.data(), &access.controls)
          : rj_arith_image_bvh_intersect_ray_w64(
                flags, state.exec_mask, 0, d, state.response_data.data(), stride, state.lane_mask,
                access.rays.data(), &access.controls);
  if (status != RJ_ARITH_SUCCESS) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (uint64_t exec_mask = state.exec_mask; exec_mask; exec_mask &= exec_mask - 1) {
    const unsigned lane = std::countr_zero(exec_mask);
    for (unsigned i = 0; i < 4; ++i) {
      cu.write_vgpr(state.dst_reg_base + i, lane, result[i][lane]);
    }
  }
}

} // namespace rocjitsu::amdgpu
