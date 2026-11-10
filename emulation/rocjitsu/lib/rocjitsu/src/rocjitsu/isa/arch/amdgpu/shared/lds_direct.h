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

/// Each primitive stores P0, P10, and P20 as three consecutive XYZW vectors.
inline void execute_lds_parameter_load(Instruction &inst, Wavefront &wf, uint32_t dst,
                                       uint32_t attribute, uint32_t component) {
  if (!wf.exec())
    return;
  if ((wf.m0() & 0x8000007f) || attribute > 32) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  const unsigned quads = wf.wf_size() / 4;
  const uint32_t starts = ((wf.m0() >> 15) | 1u) & ((1u << quads) - 1);
  const uint32_t primitives = std::popcount(starts);
  const uint64_t parameter_base = wf.lds_base() + (wf.m0() & 0xffffu);
  auto data = std::make_unique<VectorMemState>(LOCAL_MEM);
  data->wf_size = wf.wf_size();
  data->dst_reg_base = wf.vgpr_alloc().base + dst;
  data->elem_size = 4;
  data->num_elems = 1;
  data->wait_counter_type = WaitCounterType::EXPCNT;
  // Lane 3 receives zero at writeback without reading a fourth coefficient.
  data->exec_mask = whole_active_quads(wf.exec());
  data->lane_mask = data->exec_mask & 0x7777777777777777ULL;
  for (unsigned quad = 0; quad < quads; ++quad) {
    const uint32_t primitive = std::popcount(starts & ((2u << quad) - 1u)) - 1;
    const uint32_t primitive_offset = (attribute * primitives + primitive) * 12;
    for (unsigned coefficient = 0; coefficient < 3; ++coefficient)
      data->per_lane_addr[4 * quad + coefficient] =
          parameter_base + 4 * (primitive_offset + coefficient * 4 + component);
  }
  inst.set_data(std::move(data));
}

} // namespace rocjitsu::amdgpu
