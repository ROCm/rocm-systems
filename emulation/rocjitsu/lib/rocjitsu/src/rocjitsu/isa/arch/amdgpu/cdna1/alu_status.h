// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"
#include "rocjitsu/vm/amdgpu/register_access.h"

namespace rocjitsu::cdna1 {

inline void record_exceptions(amdgpu::Wavefront &wf, uint32_t causes) {
  wf.set_trapsts(wf.trapsts() | causes);
  wf.raise_alu_causes(causes);
}

// Mirror the unscaled arithmetic and the optional power-of-two scaling before
// the destination overwrites either source. Preserve the host FP environment.
template <bool Multiply, typename Inst>
uint32_t classify_binary_f32(const Inst &inst, amdgpu::Wavefront &wf, const Operand &rhs) {
  const uint64_t active = wf.exec() & wf.vgpr_write_mask();
  if (!active)
    return 0;
  const amdgpu::RegisterAccess regs(wf);
  const unsigned denorm_mode = wf.fp_denorm_mode_f32();
  amdgpu::fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  unsigned omod = amdgpu::sdwa::output_modifier<amdgpu::sdwa::ResultFormat::F32>(inst, wf);
  if constexpr (requires { inst.inst_.omod; })
    omod = amdgpu::fp_mode::effective_omod(wf.cu().arch(), denorm_mode, wf.ieee_mode(),
                                           inst.inst_.omod);
  const float scale = omod == 1 ? 2.0f : omod == 2 ? 4.0f : omod == 3 ? 0.5f : 1.0f;
  uint32_t causes = 0;
  for (uint64_t mask = active; mask; mask &= mask - 1) {
    const unsigned lane = std::countr_zero(mask);
    auto read = [&](const Operand &src, unsigned index) {
      uint32_t bits = regs.read_lane(src, lane);
      const auto magnitude = bits & 0x7fffffffu;
      if (magnitude != 0 && magnitude < 0x00800000u) {
        causes |= 1u << 1;
        if (!(denorm_mode & 1u))
          bits &= 0x80000000u;
      }
      if constexpr (requires { inst.inst_.abs; })
        bits = amdgpu::source_modifier::apply<amdgpu::fp_format::F32>(bits, index, inst.inst_.abs,
                                                                      inst.inst_.neg);
      return std::bit_cast<float>(bits);
    };
    const float lhs = read(inst.src0, 0), other = read(rhs, 1);
    volatile float result = Multiply ? lhs * other : lhs + other;
    auto flush_output = [&](float value) {
      const uint32_t raw = std::bit_cast<uint32_t>(value);
      const uint32_t magnitude = raw & amdgpu::fp_format::F32::kMagnitude;
      if ((denorm_mode & 2u) || magnitude == 0 || magnitude >= 0x00800000u)
        return value;
      causes |= (1u << 4) | (1u << 5);
      return std::bit_cast<float>(raw & amdgpu::fp_format::F32::kSign);
    };
    const float unscaled = flush_output(result);
    if (omod) {
      result = unscaled * scale;
      flush_output(result);
    }
  }
  const int flags = std::fetestexcept(FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
  causes |= (flags & FE_INVALID) ? 1u : 0u;
  causes |= (flags & FE_OVERFLOW) ? 1u << 3 : 0u;
  causes |= (flags & FE_UNDERFLOW) ? 1u << 4 : 0u;
  causes |= (flags & FE_INEXACT) ? 1u << 5 : 0u;
  return causes;
}

} // namespace rocjitsu::cdna1
