// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/cdna1/alu_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::cdna1 {

template <typename Format, typename Inst>
void record_signaling_compare(Inst &inst, amdgpu::Wavefront &wf) {
  const amdgpu::RegisterAccess regs(wf);
  for (uint64_t mask = wf.exec(); mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto is_nan = [&](const Operand &src, unsigned index) {
      typename Format::Lane bits;
      if constexpr (Format::kWidth == 16)
        bits = amdgpu::read_vop3_true16_src(src, wf, lane, amdgpu::vop3_opsel(inst.inst_), index);
      else if constexpr (Format::kWidth == 64)
        bits = regs.read_lane64(src, lane);
      else
        bits = regs.read_lane(src, lane);
      return (bits & Format::kMagnitude) > Format::kInfinity;
    };
    if (is_nan(inst.src0, 0) || is_nan(inst.src1, 1)) {
      record_exceptions(wf, 1u);
      return;
    }
  }
}

} // namespace rocjitsu::cdna1
