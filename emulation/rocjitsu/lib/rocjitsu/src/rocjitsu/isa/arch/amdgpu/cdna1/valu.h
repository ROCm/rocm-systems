// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/cdna1/alu_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::amdgpu::gfx908 {

template <typename Format, typename Inst> void record_signaling_compare(Inst &inst, Wavefront &wf) {
  const RegisterAccess regs(wf);
  for (uint64_t mask = wf.exec(); mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto is_nan = [&](const Operand &src, unsigned index) {
      typename Format::Lane bits;
      if constexpr (Format::kWidth == 16)
        bits = read_vop3_true16_src(src, wf, lane, vop3_opsel(inst.inst_), index);
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

// CDNA1 MAD/MAC rounds the FP32 product before the add and flushes denormals.
// VOP2 literal variants pass K in its arithmetic position, regardless of encoding.
template <bool Legacy, typename Inst>
void execute_mad(Inst &inst, Wavefront &wf, const Operand &dst, const Operand &a, const Operand &b,
                 const Operand &c) {
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
      return denormal::flush_value(value);
    };
    const float x = read(a, 0), y = read(b, 1), z = read(c, 2);
    volatile float product = Legacy && (x == 0.0f || y == 0.0f) ? 0.0f : x * y;
    float result = denormal::flush_value(denormal::flush_value(float(product)) + z);
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

} // namespace rocjitsu::amdgpu::gfx908
