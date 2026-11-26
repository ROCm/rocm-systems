// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 packed FP64 source carriers, guest FP controls and transactional VM stores.
#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
namespace rocjitsu::amdgpu::arith {
// A VGPR tuple contains two pairs; a scalar-backed pair replicates into both elements.
class PackedF64Input {
public:
  template <typename Operand>
  PackedF64Input(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                 uint32_t exec_mask) {
    // Resolve the base once: a legal four-register tuple may cross the encoded 256-VGPR window.
    region_ = registers.read_operand_region(operand, exec_mask, 4);
    if (region_) {
      for (int reg = 0; reg < 4; ++reg) {
        auto lanes = region_->lanes(reg);
        if (lanes.failed()) {
          data.fill(nullptr);
          return;
        }
        data[reg] = lanes.value().data();
      }
      return;
    }
    scalar_.emplace(registers, wave, operand, exec_mask);
    data = {scalar_->data[0], scalar_->data[1], scalar_->data[0], scalar_->data[1]};
  }
  std::array<const uint32_t *, 4> data;

private:
  std::optional<RegisterAccess::VgprReadRegion> region_;
  std::optional<VectorInput64> scalar_;
};
// Acquire all inputs before invoking the whole packed instruction, retaining register observers.
template <bool Fma, typename Function, typename Operand>
inline void execute_packed_f64(Wavefront &wave, Function function, uint64_t modifiers,
                               const Operand &destination, const Operand &lhs, const Operand &rhs,
                               const Operand *third = nullptr) {
  const uint32_t exec_mask = uint32_t(wave.exec());
  if (!exec_mask) {
    return;
  }
  RegisterAccess registers(wave);
  PackedF64Input a(registers, wave, lhs, exec_mask), b(registers, wave, rhs, exec_mask);
  std::optional<PackedF64Input> c;
  if constexpr (Fma) {
    c.emplace(registers, wave, *third, exec_mask);
  }
  if (wave.instruction_execution_failed()) {
    return;
  }
  const uint64_t flags = float16_f64_flags(wave, wave.cu().arithmetic_flags(), false);
  uint32_t causes = 0;
  auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    if constexpr (Fma) {
      return function(flags, uint32_t(write_mask), modifiers, d, a.data.data(), b.data.data(),
                      c->data.data(), &causes);
    } else {
      return function(flags, uint32_t(write_mask), modifiers, d, a.data.data(), b.data.data(),
                      &causes);
    }
  };
  if (auto status = registers.try_bulk_write_operand<4>(destination, exec_mask, invoke)) {
    if (*status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  // Delay observed writes until evaluation succeeds. Supported modes leave FP status unused.
  uint32_t result[4][32];
  uint32_t *d[] = {result[0], result[1], result[2], result[3]};
  const uint32_t write_mask = exec_mask & uint32_t(wave.vgpr_write_mask());
  if (invoke(d, write_mask) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  const auto reg = registers.destination_register(destination);
  if (!reg || reg->cls != RegClass::VGPR) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Preserve the same physical tuple for observed writes, including a window-crossing destination.
  auto output = registers.write_vgpr_region(wave.vgpr_alloc().base + reg->index, 4, write_mask);
  for (int word = 0; word < 4; ++word) {
    for (int lane = 0; lane < 32; ++lane) {
      if (write_mask >> lane & 1) {
        output.set_lane(word, lane, result[word][lane]);
      }
    }
  }
}
} // namespace rocjitsu::amdgpu::arith
