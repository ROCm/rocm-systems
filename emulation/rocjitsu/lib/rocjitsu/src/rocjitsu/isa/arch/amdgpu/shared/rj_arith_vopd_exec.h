// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Both slots read pre-instruction storage. Only a successful pair commits
// destinations and optional state; a rejected second slot cannot expose the
// first slot's writes or write observations. Overlapping destinations commit X
// then Y, matching the instruction's existing execution order.
template <typename Slot, typename Invoke>
inline void execute_dual_pair(Wavefront &wave, const Slot &x, const Slot &y, Invoke invoke) {
  const uint32_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  std::array<uint32_t, 32> x_result{}, y_result{};
  uint32_t causes = 0;
  if (invoke(x, registers, exec_mask, x_result.data(), &causes) != RJ_ARITH_SUCCESS ||
      invoke(y, registers, exec_mask, y_result.data(), &causes) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto x_output = registers.write_operand(*x.dst, exec_mask);
  auto y_output = registers.write_operand(*y.dst, exec_mask);
  for (unsigned lane = 0; lane < 32; lane += util::native_width_v<uint32_t>) {
    x_output.store_native(lane, util::load<uint32_t>(x_result.data() + lane), exec_mask >> lane);
    y_output.store_native(lane, util::load<uint32_t>(y_result.data() + lane), exec_mask >> lane);
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

} // namespace rocjitsu::amdgpu::arith
