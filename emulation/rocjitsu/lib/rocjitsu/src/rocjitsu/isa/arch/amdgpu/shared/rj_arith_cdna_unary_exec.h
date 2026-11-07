// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA integral unary arithmetic with staged SDWA output and true16 publication.
// The library owns arithmetic; operand observations and destination placement stay here.

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_float_exec.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Evaluate the whole wave before aliased stores. Retain legacy SDWA scaling and
// placement; old CDNA low-half results zero-extend, while high halves preserve low bits.
template <bool Half, bool OldCdna, typename Inst>
inline void execute_cdna_integral_unary(Inst &inst, Wavefront &wave, FloatUnary32Function wave32,
                                        FloatUnary64Function wave64, uint64_t flags,
                                        uint64_t modifiers, const Operand &destination,
                                        const Operand &source) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec()) {
    return;
  }
  flags = Half ? float16_flags(wave, flags, OldCdna) : float32_flags(wave, flags, OldCdna);
  RegisterAccess registers(wave);
  VectorInput32 input(registers, wave, source, wave.exec(),
                      Half ? (modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3) : 0xF);
  if (wave.instruction_execution_failed()) {
    return;
  }
  uint32_t causes = 0;
  const auto invoke = [&](uint32_t *const *d, uint64_t mask) {
    return wave.wf_size() == 32 ? wave32(flags, uint32_t(mask), modifiers, d, &input.data, &causes)
                                : wave64(flags, mask, modifiers, d, &input.data, &causes);
  };
  if constexpr (OldCdna && requires {
                  inst.sdwa_dst_sel_;
                  inst.inst_.src0;
                }) {
    if (inst.inst_.src0 == SRC_SDWA) {
      uint32_t result[64]{};
      uint32_t *d = result;
      if (invoke(&d, exec_mask) != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        if ((exec_mask >> lane) & 1) {
          // Half producers return an unscaled result; apply OMOD before placement.
          sdwa::write_lane<Half ? sdwa::ResultFormat::F16 : sdwa::ResultFormat::F32,
                           Half ? sdwa::OutputPolicy::ROUNDED : sdwa::OutputPolicy::MODE>(
              inst, wave, destination, lane, result[lane]);
        }
      }
      return;
    }
  }
  // Half outputs use a zero-initialized staging word. The byte mask controls
  // zero extension versus preserving the destination's other half.
  write_vector32_result(wave, registers, destination, exec_mask, invoke, /*allow_direct=*/!Half,
                        !Half                             ? 0xF
                        : modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC
                        : OldCdna                         ? 0xF
                                                          : 0x3);
}

// Retain SDWA modifiers and byte placement for FP32 math and exponent extraction.
// Integer exponents ignore floating output controls, as in the ISA.
template <sdwa::ResultFormat Format, sdwa::OutputPolicy Policy, typename Inst>
inline void execute_cdna_float_unary(Inst &inst, Wavefront &wave, FloatUnary32Function wave32,
                                     FloatUnary64Function wave64, uint64_t flags,
                                     uint64_t modifiers, const Operand &destination,
                                     const Operand &source, bool rdna3, bool guest_rounding,
                                     bool compatibility_all_exec = false) {
  if (inst.inst_.src0 != SRC_SDWA) {
    execute_float_unary32(wave, wave32, wave64, flags, modifiers, destination, source, rdna3,
                          guest_rounding, compatibility_all_exec);
    return;
  }
  execute_float_unary32_with_writer(
      wave, wave32, wave64, flags, modifiers, source, rdna3, guest_rounding, compatibility_all_exec,
      [&](uint64_t exec_mask, auto invoke) {
        uint32_t result[64]{};
        uint32_t *d = result;
        if (invoke(&d, exec_mask) != RJ_ARITH_SUCCESS) {
          wave.report_instruction_execution_error(
              InstructionExecutionError::UnsupportedOperandValue);
          return false;
        }
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if (((exec_mask & wave.vgpr_write_mask()) >> lane) & 1) {
            sdwa::write_lane<Format, Policy>(inst, wave, destination, lane, result[lane]);
          }
        }
        return !wave.instruction_execution_failed();
      });
}

} // namespace rocjitsu::amdgpu::arith
