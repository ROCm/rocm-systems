// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/graphics/graphics_stage.h"

#include <bit>
#include <cstring>

namespace rocjitsu::amdgpu {

void Wavefront::prepare_gs_register(VectorMemState &state, uint32_t offset, uint32_t source,
                                    uint32_t destination, bool subtract) {
  state.gs_registers = graphics_stage_ ? graphics_stage_->gs_registers() : nullptr;
  if (!state.gs_registers) {
    report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  state.gs_register_index = (offset >> 2) & 15;
  state.elem_size = state.gs_register_index < 8 ? 4 : 8;
  state.num_elems = 1;
  if (source >= num_vgprs() || destination >= num_vgprs() ||
      state.elem_size / 4 > num_vgprs() - destination) {
    report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  state.dst_reg_base = vgpr_alloc().base + destination;
  state.atomic_op = subtract ? AtomicOp::SUB : AtomicOp::ADD;
  // Only the first active lane supplies the operand and receives the return value.
  state.exec_mask = state.lane_mask = exec() & (uint64_t{0} - exec());
  state.store_data.resize(4);
  if (state.lane_mask) {
    const uint32_t operand = RegisterAccess(*this).read_vgpr(vgpr_alloc().base + source,
                                                             std::countr_zero(state.lane_mask));
    std::memcpy(state.store_data.data(), &operand, sizeof(operand));
  }
}

bool Wavefront::allocate_graphics_exports(uint32_t vertices, uint32_t primitives) {
  return graphics_stage_ && graphics_stage_->allocate_exports(*this, vertices, primitives);
}

void Wavefront::export_graphics(uint32_t target, uint32_t mask,
                                const std::array<uint32_t, 4> &sources, bool row) {
  if (status_raw() & (1u << 18))
    return;
  if (!exec()) {
    if (graphics_stage_)
      graphics_stage_->export_mask(*this, 0);
    return;
  }
  if (!graphics_stage_ || row) {
    report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    return;
  }
  for (uint32_t component = 0; component < 4; ++component)
    if ((mask & (1u << component)) && sources[component] >= num_vgprs()) {
      report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  RegisterAccess regs(*this);
  graphics_stage_->export_mask(*this, exec());
  for (uint32_t lane = 0; lane < wf_size(); ++lane) {
    if (!(exec() & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> values{};
    for (uint32_t component = 0; component < 4; ++component)
      if (mask & (1u << component))
        values[component] = regs.read_vgpr(vgpr_alloc().base + sources[component], lane);
    graphics_stage_->export_lane(*this, lane, target, mask, values);
  }
}

} // namespace rocjitsu::amdgpu
