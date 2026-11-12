// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <bit>

namespace rocjitsu::amdgpu {

/// Legacy VALU SRC0 selector 254 broadcasts a value from this workgroup's LDS.
inline uint32_t read_lds_direct_src0(const Wavefront &wf, uint32_t lane) {
  const uint32_t m0 = wf.m0();
  if (!valid_lds_direct_operand(m0)) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return 0;
  }
  const uint32_t offset = m0 & 0xffff;
  const uint32_t type = (m0 >> 16) & 7;
  const unsigned bytes = 1u << (type & 3);
  if (offset + bytes > wf.lds_size())
    return 0;
  const uint32_t address = wf.lds_base() + offset;
  if (uint64_t{address} + bytes > wf.lds().size_bytes())
    return 0;
  wf.observe_lds_direct_read(address, bytes, lane);
  if (bytes == 4)
    return wf.lds().read32(address);
  if (bytes == 2) {
    const uint16_t value = wf.lds().read16(address);
    return type == 5 ? uint32_t(int16_t(value)) : value;
  }
  const uint8_t value = wf.lds().read8(address);
  return type == 4 ? uint32_t(int8_t(value)) : value;
}

/// RDNA3+ direct LDS loads broadcast one M0-selected value to whole active quads.
inline void execute_lds_direct_load(Instruction &inst, Wavefront &wf, uint32_t dst) {
  if (!wf.exec())
    return;
  const uint32_t type = (wf.m0() >> 16) & 7, address = wf.m0() & 0xffff;
  if (!valid_lds_direct_operand(wf.m0()) ||
      (dst >= wf.num_vgprs() && wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA3_5)) {
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
