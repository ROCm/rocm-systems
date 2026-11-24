// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Remaining CDNA3 half-arithmetic status; comparison status is owned by rj-arith.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"

namespace rocjitsu::cdna3 {

enum class HalfOp { Add, Sub, Subrev, Mul };

/// IEEE causes of the FP16 arithmetic step, before the output modifier and clamp.
template <HalfOp Op>
uint32_t half_causes(uint16_t a, uint16_t b, uint32_t rounding, uint32_t denorm) {
  using F = amdgpu::fp_format::F16;
  const auto policy = amdgpu::input_denormal::Policy::make(denorm);
  a = amdgpu::input_denormal::flush_input<F>(uint32_t{a}, policy);
  b = amdgpu::input_denormal::flush_input<F>(uint32_t{b}, policy);
  if constexpr (Op == HalfOp::Sub) {
    b ^= F::kSign;
  }
  if constexpr (Op == HalfOp::Subrev) {
    a ^= F::kSign;
  }
  const auto x = a & F::kMagnitude, y = b & F::kMagnitude;
  if (x >= F::kInfinity || y >= F::kInfinity) {
    const bool snan =
        (x > F::kInfinity && !(a & F::kQuiet)) || (y > F::kInfinity && !(b & F::kQuiet));
    const bool invalid = Op == HalfOp::Mul
                             ? (x == 0 && y == F::kInfinity) || (y == 0 && x == F::kInfinity)
                             : x == F::kInfinity && y == F::kInfinity && ((a ^ b) & F::kSign);
    return snan || invalid ? RJ_ARITH_EXCEPTION_INVALID : 0;
  }
  uint32_t causes =
      (x && x < F::kMinNormal) || (y && y < F::kMinNormal) ? RJ_ARITH_EXCEPTION_INPUT_DENORM : 0;
  const double lhs = util::f16_to_f32(a), rhs = util::f16_to_f32(b);
  const double exact = Op == HalfOp::Mul ? lhs * rhs : lhs + rhs;
  // Compute the rounded significand with an unbounded exponent. Overflow can
  // produce the largest finite half under directed rounding or FP16_OVFL.
  const double magnitude = std::fabs(exact);
  const bool away = (rounding == 1 && exact > 0) || (rounding == 2 && exact < 0);
  if (magnitude >= 65536.0 || (rounding == 0 && magnitude >= 65520.0) ||
      (away && magnitude > 65504.0)) {
    return causes | RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
  }
  const uint16_t rounded =
      amdgpu::pseudo_scalar::round_f16_result(exact, rounding, 0, false, false, false);
  if (exact != util::f16_to_f32(rounded)) {
    causes |= RJ_ARITH_EXCEPTION_INEXACT;
    if ((rounded & F::kMagnitude) < F::kMinNormal) {
      causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
  }
  return causes;
}

/// Snapshot sources once so classification and numerical execution observe
/// identical values, including SDWA/DPP delegates and aliased destinations.
template <HalfOp Op, typename Execute>
void execute_half_with_status(amdgpu::Wavefront &wf, Operand &src0, Operand &src1,
                              uint32_t absolute, uint32_t negate, uint32_t opsel, Execute execute) {
  const uint64_t lanes = wf.exec() & wf.vgpr_write_mask();
  if (!lanes) {
    execute();
    return;
  }
  amdgpu::RegisterAccess regs(wf);
  amdgpu::arith::VectorInput32 a(regs, wf, src0, lanes);
  amdgpu::arith::VectorInput32 b(regs, wf, src1, lanes);
  if (wf.instruction_execution_failed()) {
    return;
  }
  StagedOperand lhs(src0, a.data, wf.wf_size());
  StagedOperand rhs(src1, b.data, wf.wf_size());
  uint32_t causes = 0;
  for (uint64_t active = lanes; active; active &= active - 1) {
    const unsigned lane = std::countr_zero(active);
    const uint16_t x = amdgpu::source_modifier::apply<amdgpu::fp_format::F16>(
        a.data[lane] >> (16 * (opsel & 1)), 0, absolute, negate);
    const uint16_t y = amdgpu::source_modifier::apply<amdgpu::fp_format::F16>(
        b.data[lane] >> (16 * ((opsel >> 1) & 1)), 1, absolute, negate);
    causes |= half_causes<Op>(x, y, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());
  }
  ScopedOperandDelegate bind_lhs(src0, &lhs);
  ScopedOperandDelegate bind_rhs(src1, &rhs);
  execute();
  if (!wf.instruction_execution_failed()) {
    wf.set_trapsts(wf.trapsts() | causes);
    wf.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::cdna3
