// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::cdna2 {

/// MI200 §3.6.1: an out-of-range destination suppresses every result of the VALU.
inline bool valu_destinations_in_range(const Instruction &inst, const amdgpu::Wavefront &wf) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto &dst = *inst.dst_operand(i);
    if (const auto reg = dst.decoded_vgpr()) {
      uint32_t index = reg->index;
      if (index < amdgpu::ACC_VGPR_OFFSET && wf.gpr_idx_en() &&
          amdgpu::gpr_idx_role_enabled(wf.gpr_idx_mode(), dst.vgpr_msb_role())) {
        index += wf.gpr_idx_offset();
        if (index >= amdgpu::ACC_VGPR_OFFSET)
          return false;
      }
      if (!wf.cdna2_vgpr_range_is_allocated(index, reg->width))
        return false;
    } else if (!dst.is_fieldless()) {
      const auto scalar = dst.to_register_ref();
      if (scalar && scalar->cls == RegClass::SGPR && scalar->index + scalar->width > wf.num_sgprs())
        return false;
    }
  }
  return true;
}

template <typename Inst>
inline void execute_with_empty_memory_mask(Inst &inst, amdgpu::Wavefront &wf) {
  // CDNA2 issues the same memory operation with zero active lanes. Let its
  // normal execution path capture that mask, addressing mode and counter.
  struct RestoreExec {
    amdgpu::Wavefront &wave;
    uint64_t saved;
    ~RestoreExec() { wave.set_exec_raw(saved); }
  } restore{wf, wf.exec_raw()};
  wf.set_exec(0);
  inst.execute_impl(wf);
}

} // namespace rocjitsu::cdna2
