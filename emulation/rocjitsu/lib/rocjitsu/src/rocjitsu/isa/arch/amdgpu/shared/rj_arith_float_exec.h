// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Translate guest denormal and IEEE controls without changing host FP controls.
inline uint64_t float32_flags(const Wavefront &wave, uint64_t flags) {
  const uint32_t denorm = wave.fp_denorm_mode_f32();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  return flags;
}

using FloatUnary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                     const uint32_t *const *, uint32_t *);
using FloatUnary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                     const uint32_t *const *, uint32_t *);

// Execute with guest FP modes and routed operands. Only successfully written
// lanes contribute fresh causes; rejected calls commit neither results nor state.
inline void execute_float_unary32(Wavefront &wave, FloatUnary32Function wave32,
                                  FloatUnary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &source, bool rdna3,
                                  bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  flags = float32_flags(wave, flags);
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, exec_mask);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, state);
      });
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

using FloatBinary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *, uint32_t *);
using FloatBinary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *, uint32_t *);

// Execute a binary operation with guest FP modes and optional accumulated state.
// All sources are acquired before writes, including when destinations alias them.
inline void execute_float_binary32(Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &lhs,
                                   const Operand &rhs, bool report_fp_status,
                                   bool rdna3_dx10_clamp = false) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  if (rdna3_dx10_clamp && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  flags = float32_flags(wave, flags) | (uint64_t(wave.fp_round_mode_f32()) << 24);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, exec_mask);
  VectorInput32 b(registers, wave, rhs, exec_mask);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
      });
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_
