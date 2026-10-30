// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Observe the quad-broadcast source lanes even when they are inactive in EXEC.
// Acquire every input before writes, and preserve the other half for FP16 P2.
// Interpolation has fixed rounding rules and no optional architectural state.
template <bool P2, bool Half, typename Wave32, typename Wave64>
inline void execute_interp(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                           uint64_t modifiers, const Operand &destination, const Operand &lhs,
                           const Operand &rhs, const Operand &third) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  const uint64_t quads =
      (exec_mask | (exec_mask >> 1) | (exec_mask >> 2) | (exec_mask >> 3)) & 0x1111111111111111ULL;
  if (Half && wave.fp16_ovfl())
    flags |= RJ_ARITH_FP16_OVFL;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, quads << (P2 ? 2 : 1),
                  Half ? (modifiers & RJ_ARITH_ALU_HIGH_A ? 0xc : 0x3) : 0xf);
  VectorInput32 b(registers, wave, rhs, exec_mask);
  VectorInput32 c(registers, wave, third, P2 ? exec_mask : quads,
                  Half && !P2 ? (modifiers & RJ_ARITH_ALU_HIGH_C ? 0xc : 0x3) : 0xf);
  write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
      },
      /*allow_direct=*/!(Half && P2),
      Half && P2 ? (modifiers & RJ_ARITH_ALU_HIGH_D ? 0xc : 0x3) : 0xf);
}

} // namespace rocjitsu::amdgpu::arith
