// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// The library's required-status mode covers nearest rounding with preserved denormals.
// Compatibility must also execute the other guest modes supported by dual MUL.
inline int gfx12_compat_dual_mul(const Wavefront &wave, uint64_t flags, uint32_t exec_mask,
                                 uint32_t *result, const uint32_t *a, const uint32_t *b,
                                 uint32_t *causes) {
  const int error = rj_arith_v_dual_mul_f32_w32_gfx12(flags, exec_mask, 0, &result, &a, &b, causes);
  if (error != RJ_ARITH_SUCCESS)
    return error;
  using F32 = fp_format::F32;
  for (uint32_t active = exec_mask; active; active &= active - 1) {
    const auto lane = std::countr_zero(active);
    const auto zero = (a[lane] ^ b[lane]) & F32::kSign;
    *causes |= fp_status::fma<F32>(a[lane], b[lane], zero, result[lane], wave.fp_round_mode_f32(),
                                   wave.fp_denorm_mode_f32());
  }
  return RJ_ARITH_SUCCESS;
}

// Both slots read pre-instruction storage. Only a successful pair commits
// destinations and optional state; a rejected second slot cannot expose the
// first slot's writes or write observations. Overlapping destinations commit X
// then Y, matching the instruction's existing execution order.
template <typename Slot, typename Invoke>
inline void execute_dual_pair(Wavefront &wave, const Slot &x, const Slot &y, Invoke invoke) {
  const uint32_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  RegisterAccess registers(wave);
  std::array<uint32_t, 32> x_result{}, y_result{};
  uint32_t causes = 0;
  if (invoke(x, registers, exec_mask, x_result.data(), &causes) != RJ_ARITH_SUCCESS ||
      invoke(y, registers, exec_mask, y_result.data(), &causes) != RJ_ARITH_SUCCESS) {
    if (!wave.instruction_execution_failed())
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  if (wave.instruction_execution_failed())
    return;
  auto x_output = registers.write_operand(*x.dst, exec_mask);
  auto y_output = registers.write_operand(*y.dst, exec_mask);
  for (unsigned lane = 0; lane < 32; lane += util::native_width_v<uint32_t>) {
    if (x_output.store_native(lane, util::load<uint32_t>(x_result.data() + lane), exec_mask >> lane)
            .failed())
      return;
    if (y_output.store_native(lane, util::load<uint32_t>(y_result.data() + lane), exec_mask >> lane)
            .failed())
      return;
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

} // namespace rocjitsu::amdgpu::arith
