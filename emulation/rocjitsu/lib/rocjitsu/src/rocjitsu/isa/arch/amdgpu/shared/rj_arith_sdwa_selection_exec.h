// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating selection forwards SDWA numerical controls to rj-arith, then places bytes.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

namespace rocjitsu::amdgpu::arith {

// Evaluate SDWA selection before publishing its selected destination bytes.
// Ordinary VOP2 and DPP delegate directly; SDWA finishes its selected result bytes.
template <int Bits, typename Inst>
inline void execute_sdwa_selection(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &lhs,
                                   const Operand &rhs, bool rdna3, bool zero_extend_low = false) {
  if (inst.inst_.src0 != SRC_SDWA) {
    if constexpr (Bits == 16) {
      execute_float_binary16(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs, rdna3,
                             zero_extend_low);
    } else {
      execute_float_binary32(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs, rdna3);
    }
    return;
  }
  if (!wave.exec()) {
    return;
  }
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec(), Bits == 16 ? 0x3 : 0xf);
  VectorInput32 b(registers, wave, rhs, wave.exec(), Bits == 16 ? 0x3 : 0xf);
  if (wave.instruction_execution_failed()) {
    return;
  }
  uint32_t result[64]{}, causes = 0;
  uint32_t *d = result;
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  flags = Bits == 16 ? float16_flags(wave, flags, rdna3) : float32_flags(wave, flags, rdna3);
  modifiers |= RJ_ARITH_ALU_SDWA | sdwa_float_modifiers(inst);
  const int status =
      wave.wf_size() == 32
          ? wave32(flags, uint32_t(exec_mask), modifiers, &d, &a.data, &b.data, &causes)
          : wave64(flags, exec_mask, modifiers, &d, &a.data, &b.data, &causes);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if ((exec_mask >> lane) & 1) {
      sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, result[lane]);
    }
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

} // namespace rocjitsu::amdgpu::arith
