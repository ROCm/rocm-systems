// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Remaining CDNA3 signaling comparisons; half-arithmetic status is owned by rj-arith.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

namespace rocjitsu::cdna3 {

/// CLAMP on GFX9 floating comparisons requests INVALID for any active NaN.
template <typename Fmt, typename Rel>
void signaling_compare(amdgpu::Wavefront &wf, const Operand &a, const Operand &b, uint32_t opsel,
                       uint32_t absolute, uint32_t negate, const Operand &dst, bool cmpx) {
  using namespace amdgpu;
  const auto denorm =
      std::is_same_v<Fmt, fp_format::F32> ? wf.fp_denorm_mode_f32() : wf.fp_denorm_mode_f16_f64();
  const auto policy = input_denormal::Policy::make(denorm);
  RegisterAccess regs(wf);
  uint32_t causes = 0;
  uint64_t result = 0;
  for (uint64_t active = wf.exec(); active; active &= active - 1) {
    const unsigned lane = std::countr_zero(active);
    auto read = [&](const Operand &src, unsigned index) -> typename Fmt::Lane {
      if constexpr (std::is_same_v<Fmt, fp_format::F64>)
        return regs.read_lane64(src, lane);
      else if constexpr (std::is_same_v<Fmt, fp_format::F16>)
        return (regs.read_lane(src, lane) >> (((opsel >> index) & 1u) * 16u)) & 0xffffu;
      else
        return regs.read_lane(src, lane);
    };
    const auto x = read(a, 0), y = read(b, 1);
    if (comparison::is_nan<Fmt>(x) || comparison::is_nan<Fmt>(y))
      causes |= RJ_ARITH_EXCEPTION_INVALID;
    if (comparison::evaluate<Fmt, Rel>(x, y, absolute, negate, policy))
      result |= uint64_t{1} << lane;
  }
  if (wf.instruction_execution_failed())
    return;
  write_wave_mask_scalar(dst, wf, result);
  if (wf.instruction_execution_failed())
    return;
  if (cmpx)
    wf.set_exec(result);
  wf.set_trapsts(wf.trapsts() | causes);
  wf.raise_alu_causes(causes);
}

} // namespace rocjitsu::cdna3
