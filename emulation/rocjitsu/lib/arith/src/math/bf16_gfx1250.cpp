// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 BF16 math retains the emulator's FP32 models and truncating narrowing.
// Output scaling precedes narrowing; strict gfx1250 empirical modes are not established.

#include "common/bits.h"
#include "common/validation.h"
#include "math/exp_model.h"
#include "math/log_model.h"
#include "math/rcp_model.h"
#include "math/rsq_model.h"
#include "math/sqrt_model.h"
#include "math/trig_model.h"
#include "math/x86_64/bf16_math.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {
using namespace rocjitsu::arith;
enum class Operation { Rcp, Rsq, Sqrt, Exp, Log, Sin, Cos, Tanh };

// Evaluate the inherited FP32 model, including its NaN and denormal policy.
template <Operation Op> uint32_t evaluate(uint32_t bits) {
  if constexpr (Op == Operation::Rcp) {
    return rcp_f32_bits(bits, true);
  } else if constexpr (Op == Operation::Rsq) {
    return rsq_f32_bits(bits, true);
  } else if constexpr (Op == Operation::Sqrt) {
    return sqrt_f32_bits(bits, true);
  } else if constexpr (Op == Operation::Exp) {
    return exp_f32_bits(bits, true);
  } else if constexpr (Op == Operation::Log) {
    return log_f32_bits(bits, true);
  } else if constexpr (Op == Operation::Sin || Op == Operation::Cos) {
    return trig::evaluate(bits, Op == Operation::Cos, 3, true);
  } else {
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
      return bits | 0x400000U;
    }
    return as_bits(__builtin_tanhf(as_float(bits)));
  }
}

// Select an existing FP32 numeric core without its output modifiers or status reporting.
template <Operation Op> constexpr Bf16MathCore fp32_core() {
  if constexpr (Op == Operation::Rcp) {
    return rj_arith_v_rcp_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Rsq) {
    return rj_arith_v_rsq_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Sqrt) {
    return rj_arith_v_sqrt_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Exp) {
    return rj_arith_v_exp_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Log) {
    return rj_arith_v_log_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Sin) {
    return rj_arith_v_sin_f32_w32_gfx12;
  }
  if constexpr (Op == Operation::Cos) {
    return rj_arith_v_cos_f32_w32_gfx12;
  }
  return nullptr;
}

// Scale and clamp in FP32, then truncate to BF16 and finalize tiny OMOD results.
uint16_t narrow(uint32_t bits, uint32_t mode) {
  const int omod = int((mode & RJ_ARITH_ALU_OMOD_MASK) >> 6);
  if (omod) {
    const float scale = omod == 3 ? 0.5f : omod == 2 ? 4.0f : 2.0f;
    bits = as_bits(as_float(bits) * scale);
    if ((bits & 0x7fffffffU) < 0x800000U) {
      bits = 0;
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    if ((bits & 0x80000000U) || (bits & 0x7fffffffU) > 0x7f800000U) {
      bits = 0;
    } else if (bits > 0x3f800000U) {
      bits = 0x3f800000U;
    }
  }
  uint16_t result = uint16_t(bits >> 16);
  if (omod && (result & 0x7fffU) < 0x80U) {
    result = 0;
  }
  return result;
}

// Evaluate active lanes with whole-wave alias safety and selected-half writes.
template <Operation Op>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t controls = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_HIGH_A |
                                RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  constexpr uint64_t fp = RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE |
                          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (int error = validate(flags, mode & ~controls, fp)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Op != Operation::Tanh) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      return bf16_math_x86_64_v4(fp32_core<Op>(), flags, exec_mask, uint32_t(mode), d[0], a[0]);
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      return bf16_math_x86_64_v3(fp32_core<Op>(), flags, exec_mask, uint32_t(mode), d[0], a[0]);
    }
#endif
  }
  const int source_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int destination_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32]{};
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask & (1U << lane)) {
      uint32_t bits = (a[0][lane] >> source_shift) << 16;
      if (mode & RJ_ARITH_ALU_ABS_A) {
        bits &= 0x7fffffffU;
      }
      if (mode & RJ_ARITH_ALU_NEG_A) {
        bits ^= 0x80000000U;
      }
      result[lane] = narrow(evaluate<Op>(bits), uint32_t(mode));
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask & (1U << lane)) {
      const uint32_t keep = ~(0xffffU << destination_shift);
      d[0][lane] = (d[0][lane] & keep) | (uint32_t(result[lane]) << destination_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate inherited BF16 rcp arithmetic; legacy execution produces no FP status.
int rj_arith_v_rcp_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Rcp>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 rsq arithmetic; legacy execution produces no FP status.
int rj_arith_v_rsq_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Rsq>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 sqrt arithmetic; legacy execution produces no FP status.
int rj_arith_v_sqrt_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t * /*excp_flag_user*/) {
  return run<Operation::Sqrt>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 exp arithmetic; legacy execution produces no FP status.
int rj_arith_v_exp_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Exp>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 log arithmetic; legacy execution produces no FP status.
int rj_arith_v_log_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Log>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 sin arithmetic; legacy execution produces no FP status.
int rj_arith_v_sin_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Sin>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 cos arithmetic; legacy execution produces no FP status.
int rj_arith_v_cos_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<Operation::Cos>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate inherited BF16 tanh arithmetic; legacy execution produces no FP status.
int rj_arith_v_tanh_bf16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t * /*excp_flag_user*/) {
  return run<Operation::Tanh>(flags, exec_mask, instruction_flags, d, a);
}
