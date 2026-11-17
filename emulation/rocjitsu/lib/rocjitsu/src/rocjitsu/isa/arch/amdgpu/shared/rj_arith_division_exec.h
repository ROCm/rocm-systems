// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu::arith {

enum class DivisionOperation { Scale, Fmas, Fixup };

// SCALE produces SDST after its vector result. CDNA preserves inactive bits from VCC;
// the RDNA policy clears them.
// FMAS consumes pre-instruction VCC. FMAS/FIXUP optionally accumulate
// architectural exception state. All sources are acquired before any destination is written.
template <unsigned Width, DivisionOperation Op, typename Wave32, typename Wave64,
          typename CommitCondition = std::nullptr_t>
inline void execute_division(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                             uint64_t modifiers, const Operand &destination, const Operand &lhs,
                             const Operand &rhs, const Operand &third, bool rdna3,
                             CommitCondition commit_condition = nullptr,
                             bool preserve_inactive_condition = false) {
  const uint64_t exec_mask = wave.exec();
  const uint64_t write_mask = exec_mask & wave.vgpr_write_mask();
  const uint64_t old_condition = wave.vcc();
  flags = Width == 64 ? float16_f64_flags(wave, flags, rdna3) : float32_flags(wave, flags, rdna3);
  if (!exec_mask) {
    if constexpr (Op == DivisionOperation::Scale) {
      uint32_t narrow_condition = 0;
      uint64_t condition = 0;
      const int error =
          wave.wf_size() == 32
              ? wave32(flags, 0, modifiers, nullptr, &narrow_condition, nullptr, nullptr, nullptr)
              : wave64(flags, 0, modifiers, nullptr, &condition, nullptr, nullptr, nullptr);
      if (error != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      else
        commit_condition(preserve_inactive_condition ? old_condition
                         : wave.wf_size() == 32      ? narrow_condition
                                                     : condition);
    }
    return;
  }
  RegisterAccess registers(wave);
  using Input = std::conditional_t<Width == 64, VectorInput64, VectorInput32>;
  Input a(registers, wave, lhs, exec_mask), b(registers, wave, rhs, exec_mask),
      c(registers, wave, third, exec_mask);
  const auto words = [](Input &input) -> const uint32_t *const * {
    if constexpr (Width == 64)
      return input.data.data();
    else
      return &input.data;
  };
  uint64_t condition = 0;
  uint32_t causes = 0;
  const auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    if constexpr (Op == DivisionOperation::Scale) {
      if (wave.wf_size() == 32) {
        uint32_t narrow_condition = 0;
        const int error = wave32(flags, uint32_t(write_mask), modifiers, d, &narrow_condition,
                                 words(a), words(b), words(c));
        condition = narrow_condition;
        return error;
      }
      return wave64(flags, write_mask, modifiers, d, &condition, words(a), words(b), words(c));
    } else {
      uint32_t *state = &causes;
      if constexpr (Op == DivisionOperation::Fixup)
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, words(a), words(b), words(c),
                            state)
                   : wave64(flags, write_mask, modifiers, d, words(a), words(b), words(c), state);
      else
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d, words(a),
                                             words(b), words(c), uint32_t(old_condition), state)
                                    : wave64(flags, write_mask, modifiers, d, words(a), words(b),
                                             words(c), old_condition, state);
    }
  };
  bool success;
  if (write_mask != exec_mask) {
    // DPP masks VGPR stores, but SDST and FP status still cover every EXEC lane.
    std::array<std::array<uint32_t, 64>, Width / 32> result;
    std::array<uint32_t *, Width / 32> d;
    for (unsigned reg = 0; reg < Width / 32; ++reg) {
      d[reg] = result[reg].data();
    }
    success =
        !wave.instruction_execution_failed() && invoke(d.data(), exec_mask) == RJ_ARITH_SUCCESS;
    if (!success) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    } else if (write_mask) {
      auto publish = [&](uint32_t *const *output, uint64_t lanes) {
        for (unsigned reg = 0; reg < Width / 32; ++reg) {
          for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
            if ((lanes >> lane) & 1) {
              output[reg][lane] = result[reg][lane];
            }
          }
        }
        return RJ_ARITH_SUCCESS;
      };
      if constexpr (Width == 64)
        success = write_vector64_result(wave, registers, destination, write_mask, publish);
      else
        success = write_vector32_result(wave, registers, destination, write_mask, publish);
    }
  } else if constexpr (Width == 64)
    success = write_vector64_result(wave, registers, destination, write_mask, invoke);
  else
    success = write_vector32_result(wave, registers, destination, write_mask, invoke);
  if (!success)
    return;
  if constexpr (Op == DivisionOperation::Scale)
    commit_condition(condition | (preserve_inactive_condition ? old_condition & ~exec_mask : 0));
  else {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::amdgpu::arith
