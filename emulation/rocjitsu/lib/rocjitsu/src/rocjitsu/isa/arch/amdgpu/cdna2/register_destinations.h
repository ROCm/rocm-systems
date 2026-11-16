// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"

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
      if (!wf.unified_vgpr_range_is_allocated(index, reg->width))
        return false;
    } else if (!dst.is_fieldless()) {
      const auto scalar = dst.to_register_ref();
      if (scalar && scalar->cls == RegClass::SGPR &&
          wf.exceeds_rounded_sgpr_allocation(scalar->index, scalar->width))
        return false;
    }
  }
  return true;
}

} // namespace rocjitsu::cdna2
