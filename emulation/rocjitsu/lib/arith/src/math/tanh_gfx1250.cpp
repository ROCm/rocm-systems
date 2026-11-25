// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 TANH retains the emulator's host-libm core and rounded output modifiers.
// The CPU FP environment is assumed default and is never changed here.

#include "common/bits.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/float_output.h"
#include "math/x86_64/tanh_loose.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <stdint.h>

namespace {
using namespace rocjitsu::arith;

// Evaluate the legacy FP32 TANH core, explicitly quieting input NaNs.
uint32_t tanh_bits(uint32_t bits) {
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return bits | 0x400000U;
  }
  return as_bits(__builtin_tanhf(as_float(bits)));
}

// Apply TANH then destination-format OMOD/CLAMP, preserving aliases and inactive lanes.
template <bool Half>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t controls = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK |
                                RJ_ARITH_ALU_CLAMP |
                                (Half ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0);
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
  uint32_t result[32]{};
  for (int lane = 0; lane < 32; ++lane) {
    if (!(exec_mask & (1U << lane))) {
      continue;
    }
    uint32_t bits = a[0][lane];
    if constexpr (Half) {
      bits = as_bits(f16_to_float(uint16_t(bits >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0))));
    }
    if (mode & RJ_ARITH_ALU_ABS_A) {
      bits &= 0x7fffffffU;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      bits ^= 0x80000000U;
    }
    result[lane] = bits;
  }
  bool vectorized = false;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (!compatibility && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    tanh_loose_x86_64_v4(result, result);
    vectorized = true;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!compatibility && !vectorized && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    tanh_loose_x86_64_v3(result, result);
    vectorized = true;
  }
#endif
  (void)compatibility;
  for (int lane = 0; lane < 32; ++lane) {
    if (!(exec_mask & (1U << lane))) {
      continue;
    }
    uint32_t bits = vectorized ? result[lane] : tanh_bits(result[lane]);
    if constexpr (Half) {
      // Legacy VOP3 narrows to nearest before scaling the architectural half.
      bits = float_to_f16(as_float(bits), flags & RJ_ARITH_FP16_OVFL);
      result[lane] = selection_output<16>(bits, uint32_t(mode), flags);
    } else {
      result[lane] = selection_output<32>(bits, uint32_t(mode), flags);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (exec_mask & (1U << lane)) {
      if constexpr (Half) {
        const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        d[0][lane] = (d[0][lane] & ~(0xffffU << shift)) | (result[lane] << shift);
      } else {
        d[0][lane] = result[lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate FP32 TANH with the inherited libm model and no FP-status reporting.
int rj_arith_v_tanh_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<false>(flags, exec_mask, instruction_flags, d, a);
}

// Evaluate FP16 TANH, narrowing before output modifiers and preserving the unselected half.
int rj_arith_v_tanh_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t * /*excp_flag_user*/) {
  return run<true>(flags, exec_mask, instruction_flags, d, a);
}
