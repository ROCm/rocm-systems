// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna2/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace cdna2 {

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                               amdgpu::ScalarMemState *state) {
  return amdgpu::addr_calc::smem_calculate_address(inst, wf, state, true);
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::flat_calculate_addresses<amdgpu::addr_calc::ScratchAddressMode::LegacyCdna>(
      inst, wf, d);
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  // The LDS-to-buffer instruction sources M0 and EXEC. Its fixed LDS bit
  // and lack of a VADDR operand distinguish it from ordinary buffer stores.
  if (inst.op == kBufferStoreLdsDwordMubuf && (!inst.lds || inst.idxen || inst.offen)) {
    wf.report_instruction_execution_error(
        amdgpu::InstructionExecutionError::InvalidInstructionEncoding);
    amdgpu::reject_vector_memory_access(d);
    return;
  }
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, d);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, d);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace cdna2
} // namespace rocjitsu
