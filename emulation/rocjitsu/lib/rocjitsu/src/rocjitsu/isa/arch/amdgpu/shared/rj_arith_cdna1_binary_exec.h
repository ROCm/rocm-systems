// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx908 binary arithmetic passes SDWA floating modifiers to rj-arith so its
// inherited exception classifier sees scaling; only destination placement remains here.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

namespace rocjitsu::amdgpu::arith {

// Capture selected sources, evaluate arithmetic/status, then publish destination bytes.
// Ordinary and DPP forms keep direct bulk stores; SDWA stages its selected output.
template <typename Inst>
inline void execute_cdna1_binary32(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &lhs,
                                   const Operand &rhs, bool rdna3, bool compatibility_mul) {
  if (inst.inst_.src0 != SRC_SDWA) {
    execute_float_binary32(wave, wave32, wave64, flags, modifiers, destination, lhs, rhs, rdna3,
                           compatibility_mul);
    return;
  }
  if (!wave.exec()) {
    return;
  }
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec());
  VectorInput32 b(registers, wave, rhs, wave.exec());
  if (wave.instruction_execution_failed()) {
    return;
  }
  modifiers |=
      uint64_t(inst.sdwa_omod_) * RJ_ARITH_ALU_OMOD_2 | (inst.sdwa_clamp_ ? RJ_ARITH_ALU_CLAMP : 0);
  flags = float32_flags(wave, flags, rdna3);
  modifiers |= RJ_ARITH_ALU_SDWA;
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  uint32_t results[64]{}, causes = 0;
  uint32_t *d = results;
  const int error =
      wave.wf_size() == 32
          ? wave32(flags, uint32_t(exec_mask), modifiers, &d, &a.data, &b.data, &causes)
          : wave64(flags, exec_mask, modifiers, &d, &a.data, &b.data, &causes);
  if (error != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if ((exec_mask >> lane) & 1) {
      sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, results[lane]);
    }
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

} // namespace rocjitsu::amdgpu::arith
