// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_COMPARE_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_COMPARE_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

using Compare32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *, const uint32_t *const *,
                                  const uint32_t *const *);
using Compare64Function = int (*)(uint64_t, uint64_t, uint64_t, uint64_t *, const uint32_t *const *,
                                  const uint32_t *const *);

// Keep observed source views alive through the call. Constant predicates and
// empty EXEC do not observe inputs; rj-arith still validates flags in those cases.
template <unsigned Bits, unsigned SecondBits = Bits, typename Invoke>
inline int with_compare_inputs(Wavefront &wave, uint64_t exec_mask, const Operand &source_a,
                               const Operand &source_b, bool constant, Invoke invoke) {
  static_assert(SecondBits == Bits || SecondBits == 32);
  if (constant || !exec_mask)
    return invoke(nullptr, nullptr);
  RegisterAccess registers(wave);
  if constexpr (Bits == 64) {
    VectorInput64 a(registers, wave, source_a, exec_mask);
    if constexpr (SecondBits == 64) {
      VectorInput64 b(registers, wave, source_b, exec_mask);
      return invoke(a.data.data(), b.data.data());
    } else {
      VectorInput32 b(registers, wave, source_b, exec_mask);
      return invoke(a.data.data(), &b.data);
    }
  } else {
    VectorInput32 a(registers, wave, source_a, exec_mask), b(registers, wave, source_b, exec_mask);
    return invoke(&a.data, &b.data);
  }
}

// Commit receives a wave-sized mask only after a successful call. Empty EXEC
// still produces a zero mask; constant predicates never observe source registers.
// CLASS predicates use a one-word class mask even when A spans two VGPRs.
template <unsigned Bits, unsigned SecondBits = Bits, typename Commit>
inline void execute_compare(Wavefront &wave, Compare32Function wave32, Compare64Function wave64,
                            uint64_t flags, uint64_t modifiers, const Operand &source_a,
                            const Operand &source_b, bool constant, Commit commit) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  uint64_t result = 0;
  auto invoke = [&](const uint32_t *const *a, const uint32_t *const *b) {
    if (wave.wf_size() == 64)
      return wave64(flags, exec_mask, modifiers, &result, a, b);
    uint32_t result32 = 0;
    const int status = wave32(flags, uint32_t(exec_mask), modifiers, &result32, a, b);
    result = result32;
    return status;
  };
  const int status =
      with_compare_inputs<Bits, SecondBits>(wave, exec_mask, source_a, source_b, constant, invoke);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  commit(result);
}

using FloatCompare32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *,
                                       const uint32_t *const *, const uint32_t *const *,
                                       uint32_t *);
using FloatCompare64Function = int (*)(uint64_t, uint64_t, uint64_t, uint64_t *,
                                       const uint32_t *const *, const uint32_t *const *,
                                       uint32_t *);

// Report fresh causes separately from sticky state so repeated exceptions still
// reach trap delivery. A rejected call commits neither its mask nor its causes.
template <unsigned Bits, typename Commit>
inline void execute_float_compare(Wavefront &wave, FloatCompare32Function wave32,
                                  FloatCompare64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &source_a, const Operand &source_b, bool constant,
                                  bool report_fp_status, Commit commit,
                                  uint64_t exec_mask = ~0ULL) {
  const uint32_t denorm = Bits == 32 ? wave.fp_denorm_mode_f32() : wave.fp_denorm_mode_f16_f64();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  exec_mask &= wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  uint64_t result = 0;
  uint32_t causes = 0;
  auto invoke = [&](const uint32_t *const *a, const uint32_t *const *b) {
    uint32_t *state = &causes;
    if (wave.wf_size() == 64)
      return wave64(flags, exec_mask, modifiers, &result, a, b, state);
    uint32_t result32 = 0;
    const int status = wave32(flags, uint32_t(exec_mask), modifiers, &result32, a, b, state);
    result = result32;
    return status;
  };
  const int status =
      with_compare_inputs<Bits>(wave, exec_mask, source_a, source_b, constant, invoke);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  commit(result);
  if (report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_COMPARE_EXEC_H_
