// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 SDWA binary arithmetic stages library-owned output before byte placement.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

namespace rocjitsu::amdgpu::arith {

// Evaluate through the common binary/status path before publishing SDWA-selected bytes. All source
// reads precede writes, including overlapping registers.
template <typename Inst>
inline void execute_sdwa_binary32(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                  FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &lhs,
                                  const Operand &rhs, bool rdna3, bool compatibility_mul) {
  if (inst.inst_.src0 != SRC_SDWA) {
    execute_float_binary32(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs, rdna3,
                           compatibility_mul);
    return;
  }
  modifiers |= RJ_ARITH_ALU_SDWA | sdwa_float_modifiers(inst);
  execute_float_binary32_with_writer(
      wave, wave32, wave64, flags, modifiers, lhs, rhs, rdna3, compatibility_mul, [&](auto invoke) {
        uint32_t results[64]{}, *d = results;
        const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
        if (invoke(&d, exec_mask) != RJ_ARITH_SUCCESS) {
          wave.report_instruction_execution_error(
              InstructionExecutionError::UnsupportedOperandValue);
          return false;
        }
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if ((exec_mask >> lane) & 1) {
            sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane,
                                                       results[lane]);
          }
        }
        return !wave.instruction_execution_failed();
      });
}

} // namespace rocjitsu::amdgpu::arith
