// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP32 binary arithmetic retains SDWA output scaling, clamp and byte placement.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

namespace rocjitsu::amdgpu::arith {

// Stage SDWA results before destination transforms so aliasing and observers see
// the same source reads and selected destination bytes as ordinary execution.
template <typename Inst>
inline void execute_cdna_binary32(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                  FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &lhs,
                                  const Operand &rhs, bool rdna3, bool compatibility_mul) {
  if (inst.inst_.src0 != SRC_SDWA) {
    execute_float_binary32(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs, rdna3,
                           compatibility_mul);
    return;
  }
  execute_float_binary32_with_writer(
      wave, wave32, wave64, flags, modifiers, lhs, rhs, rdna3, compatibility_mul, [&](auto invoke) {
        uint32_t result[64]{};
        uint32_t *d = result;
        const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
        if (invoke(&d, exec_mask) != RJ_ARITH_SUCCESS) {
          wave.report_instruction_execution_error(
              InstructionExecutionError::UnsupportedOperandValue);
          return false;
        }
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if ((exec_mask >> lane) & 1) {
            sdwa::write_lane<sdwa::ResultFormat::F32>(inst, wave, destination, lane, result[lane]);
          }
        }
        return !wave.instruction_execution_failed();
      });
}

} // namespace rocjitsu::amdgpu::arith
