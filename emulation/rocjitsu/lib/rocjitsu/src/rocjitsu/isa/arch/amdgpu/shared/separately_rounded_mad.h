// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::amdgpu {

// CDNA1 and RDNA1 MAD/MAC round the FP32 product before the add and flushes denormals.
// VOP2 literal variants pass K in its arithmetic position, regardless of encoding.
template <bool Legacy, typename Inst>
void execute_separately_rounded_mad(Inst &inst, Wavefront &wf, const Operand &dst, const Operand &a,
                                    const Operand &b, const Operand &c) {
  const uint64_t active = dpp::execution_lane_mask(inst, wf);
  const RegisterAccess regs(wf);
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(active & (uint64_t{1} << lane)))
      continue;
    auto read = [&](const Operand &source, unsigned index) {
      float value = std::bit_cast<float>(regs.read_lane(source, lane));
      if constexpr (requires { inst.inst_.abs; }) {
        // MAC always takes its accumulator directly from the destination.
        if (&source != &dst)
          value = source_modifier::apply_to_float(value, index, inst.inst_.abs, inst.inst_.neg);
      }
      return value;
    };
    const float x = read(a, 0), y = read(b, 1), z = read(c, 2);
    float result = separately_rounded_mad_f32<Legacy>(x, y, z);
    if constexpr (requires { inst.inst_.omod; }) {
      const auto omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(),
                                                wf.ieee_mode(), inst.inst_.omod);
      result = fp_mode::apply_omod_f32(result, omod, wf.fp_round_mode_f32());
      if (inst.inst_.clamp)
        result = clamp_floating_result(result, wf);
    }
    sdwa::write_lane<sdwa::ResultFormat::F32>(inst, wf, dst, lane, std::bit_cast<uint32_t>(result));
  }
}

} // namespace rocjitsu::amdgpu
