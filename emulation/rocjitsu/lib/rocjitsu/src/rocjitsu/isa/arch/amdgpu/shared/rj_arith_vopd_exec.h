// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/detail/rj_arith_dual.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

inline bool commit_dual_word(RegisterAccess &registers, const Operand &destination,
                             uint32_t write_mask, const uint32_t *result) {
  auto output = registers.write_operand(destination, write_mask);
  for (unsigned lane = 0; lane < 32; lane += util::register_block_lanes_v<uint32_t>) {
    if (output.store_native(lane, util::load<uint32_t>(result + lane), write_mask >> lane)
            .failed()) {
      return false;
    }
  }
  return true;
}

// Two moves of inline zero have identical inputs and status. Evaluate once,
// then commit the same result in slot order. No source registers are read.
inline void execute_dual_zero_moves(Wavefront &wave, const Operand &x, const Operand &y) {
  const uint32_t exec = wave.exec();
  if (!exec || wave.instruction_execution_failed()) {
    return;
  }
  alignas(64) static constexpr std::array<uint32_t, 32> zero{};
  alignas(64) std::array<uint32_t, 32> result{};
  const uint32_t *src = zero.data();
  uint32_t *dst = result.data();
  if (rj_arith_v_dual_mov_b32_w32(wave.cu().arithmetic_flags(), exec, 0, &dst, &src) !=
      RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  RegisterAccess registers(wave);
  const uint32_t write_mask = exec & wave.vgpr_write_mask();
  if (commit_dual_word(registers, x, write_mask, dst)) {
    commit_dual_word(registers, y, write_mask, dst);
  }
}

// Both slots read pre-instruction storage. Only a successful pair commits
// destinations and optional state; a rejected second slot cannot expose the
// first slot's writes or write observations. Overlapping destinations commit X
// then Y, matching the instruction's existing execution order.
template <int ResultWords = 1, typename Slot, typename Invoke>
inline void execute_dual_pair(Wavefront &wave, const Slot &x, const Slot &y, Invoke invoke) {
  const uint32_t exec_mask = wave.exec();
  const uint32_t write_mask = exec_mask & wave.vgpr_write_mask();
  if (!wave.exec()) {
    return;
  }
  RegisterAccess registers(wave);
  static_assert(ResultWords == 1 || ResultWords == 2);
  std::array<uint32_t, 32 * ResultWords> x_result{}, y_result{};
  uint32_t causes = 0;
  if (invoke(x, registers, exec_mask, x_result.data(), &causes) != RJ_ARITH_SUCCESS ||
      invoke(y, registers, exec_mask, y_result.data(), &causes) != RJ_ARITH_SUCCESS) {
    if (!wave.instruction_execution_failed()) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  if (wave.instruction_execution_failed()) {
    return;
  }
  // rj-arith stores register pairs as two lane arrays. Use the observer-aware
  // wide destination view for FP64 slots; ordinary RDNA slots retain 32-bit stores.
  auto commit = [&](const Slot &slot, const auto &result) {
    if constexpr (ResultWords == 2) {
      if (slot.dst->size_bits() == 64) {
        auto output = registers.write_operand64(*slot.dst, write_mask);
        for (unsigned lane = 0; lane < 32; lane += util::register_block_lanes_v<uint64_t>) {
          if (output
                  .store_native(
                      lane, util::load64<uint64_t>(result.data() + lane, result.data() + 32 + lane),
                      write_mask >> lane)
                  .failed()) {
            return false;
          }
        }
        return true;
      }
    }
    return commit_dual_word(registers, *slot.dst, write_mask, result.data());
  };
  if (!commit(x, x_result) || !commit(y, y_result)) {
    return;
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

} // namespace rocjitsu::amdgpu::arith
