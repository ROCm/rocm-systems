// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older legacy-MUL bindings, retaining SDWA scaling and destination placement.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

namespace rocjitsu::amdgpu::arith {

// Capture both sources before writing an aliased destination. For SDWA, the
// arithmetic API applies scaling/clamp and the final store only places bytes.
template <typename Inst>
inline void execute_legacy_multiply32(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                      FloatBinary64Function wave64, uint64_t flags,
                                      uint64_t modifiers, const Operand &destination,
                                      const Operand &lhs, const Operand &rhs) {
  if constexpr (requires { inst.sdwa_dst_sel_; }) {
    if (inst.inst_.src0 == SRC_SDWA) {
      const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
      if (!wave.exec()) {
        return;
      }
      RegisterAccess registers(wave);
      VectorInput32 a(registers, wave, lhs, wave.exec());
      VectorInput32 b(registers, wave, rhs, wave.exec());
      if (wave.instruction_execution_failed()) {
        return;
      }
      flags = float32_flags(wave, flags, true);
      modifiers |= RJ_ARITH_ALU_SDWA;
      // The arithmetic entry point applies the guest-mode OMOD eligibility rule.
      if constexpr (requires { inst.sdwa_omod_; }) {
        modifiers |= uint64_t(inst.sdwa_omod_) * RJ_ARITH_ALU_OMOD_2;
      }
      modifiers |= inst.sdwa_clamp_ ? RJ_ARITH_ALU_CLAMP : 0;
      uint32_t result[64]{}, *d = result, state = 0;
      const int error =
          wave.wf_size() == 32
              ? wave32(flags, uint32_t(exec_mask), modifiers, &d, &a.data, &b.data, &state)
              : wave64(flags, exec_mask, modifiers, &d, &a.data, &b.data, &state);
      if (error != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        if ((exec_mask >> lane) & 1) {
          sdwa::write_lane(inst, wave, destination, lane, result[lane]);
        }
      }
      return;
    }
  }
  execute_float_binary32(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs,
                         /*rdna3=*/true, /*compatibility_mul=*/false);
}

} // namespace rocjitsu::amdgpu::arith
