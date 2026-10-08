// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

/// RDNA3+ direct LDS loads broadcast one M0-selected value to whole active quads.
inline void execute_lds_direct_load(Instruction &inst, Wavefront &wf, uint32_t dst) {
  if (!wf.exec())
    return;
  const uint32_t type = (wf.m0() >> 16) & 7, address = wf.m0() & 0xffff;
  if (!valid_lds_direct_operand(wf.m0()) || dst >= wf.num_vgprs()) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto d = std::make_unique<VectorMemState>(LOCAL_MEM);
  d->wf_size = wf.wf_size();
  d->dst_reg_base = wf.vgpr_alloc().base + dst;
  d->elem_size = type == 2 ? 4 : (type & 1 ? 2 : 1);
  d->num_elems = 1;
  d->sign_extend = type & 4;
  d->wait_counter_type = WaitCounterType::EXPCNT;
  d->exec_mask = whole_active_quads(wf.exec());
  d->lane_mask = d->exec_mask;
  d->per_lane_addr.fill(wf.lds_base() + address);
  inst.set_data(std::move(d));
}

} // namespace rocjitsu::amdgpu
