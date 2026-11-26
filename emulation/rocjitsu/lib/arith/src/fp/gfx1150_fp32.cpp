// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1150 FP32 arithmetic retains its inherited Compatibility exception policy.
// Numeric work uses the existing scalar/SIMD APIs; status precedes CLAMP.

#include "common/alu.h"
#include "common/validation.h"
#include "fp/compatibility_fma_status.h"
#include "fp/fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace {

using namespace rocjitsu::arith;

enum class Operation { Add, Sub, Subrev, Mul, Fma };

// Call the inherited numeric model with the requested wave width and controls.
template <Operation Op, int Width>
int numeric(uint64_t flags, uint64_t exec_mask, uint64_t modifiers, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Width == 32) {
    if constexpr (Op == Operation::Add) {
      return rj_arith_v_add_f32_w32(flags, uint32_t(exec_mask), modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Sub) {
      return rj_arith_v_sub_f32_w32(flags, uint32_t(exec_mask), modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Subrev) {
      return rj_arith_v_subrev_f32_w32(flags, uint32_t(exec_mask), modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Mul) {
      return rj_arith_v_mul_f32_w32(flags, uint32_t(exec_mask), modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Fma) {
      return rj_arith_v_fma_f32_w32(flags, uint32_t(exec_mask), modifiers, d, a, b, c, nullptr);
    }
  } else {
    if constexpr (Op == Operation::Add) {
      return rj_arith_v_add_f32_w64(flags, exec_mask, modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Sub) {
      return rj_arith_v_sub_f32_w64(flags, exec_mask, modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Subrev) {
      return rj_arith_v_subrev_f32_w64(flags, exec_mask, modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Mul) {
      return rj_arith_v_mul_f32_w64(flags, exec_mask, modifiers, d, a, b, nullptr);
    }
    if constexpr (Op == Operation::Fma) {
      return rj_arith_v_fma_f32_w64(flags, exec_mask, modifiers, d, a, b, c, nullptr);
    }
  }
}

// True only for the inherited gfx1150 reporting policy; other modes leave state unused.
bool reporting(uint64_t flags) {
  return (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY &&
         (flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
}

// Classify the same modified FMA triples as rdna3_5/valu.h, before destination CLAMP.
template <Operation Op>
uint32_t causes(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, const uint32_t *a,
                const uint32_t *b, const uint32_t *c, const uint32_t *results) {
  const unsigned omod = (modifiers >> 6) & 3;
  uint32_t status = 0;
  for (int lane = 0; lane < 32; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint32_t x = alu_input_bits(a[lane], modifiers), y = alu_input_bits(b[lane], modifiers >> 1), z;
    if constexpr (Op == Operation::Fma) {
      z = alu_input_bits(c[lane], modifiers >> 2);
    } else if constexpr (Op == Operation::Mul) {
      z = (x ^ y) & 0x80000000U;
    } else {
      if constexpr (Op == Operation::Sub) {
        y ^= 0x80000000U;
      }
      if constexpr (Op == Operation::Subrev) {
        x ^= 0x80000000U;
      }
      z = y;
      y = 0x3f800000U;
    }
    status |= compatibility_fma_causes<false>(flags, x, y, z, results[lane]);
    if (omod) {
      const uint32_t scale = omod == 1 ? 0x40000000U : omod == 2 ? 0x40800000U : 0x3f000000U;
      const uint32_t zero = results[lane] & 0x80000000U;
      const uint32_t scaled = fma_instruction_value(
          results[lane], scale, zero, flags | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0, true, false);
      status |= compatibility_fma_causes<false>(flags, results[lane], scale, zero, scaled);
    }
  }
  return status;
}

// Evaluate one wave half. Return the effective store mask for staged Wave64 calls.
template <Operation Op>
int run32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
          uint32_t *trapsts, uint32_t &write_mask) {
  write_mask = 0;
  if (modifiers >> 32) {
    return execute_dpp(flags, exec_mask, modifiers, a,
                       [&](uint32_t exec_mask, const uint32_t *const *source) {
                         return run32<Op>(flags, exec_mask, uint32_t(modifiers), d, source, b, c,
                                          trapsts, write_mask);
                       });
  }
  const uint64_t numeric_flags =
      resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (!reporting(flags)) {
    const int error = numeric<Op, 32>(numeric_flags, exec_mask, modifiers, d, a, b, c);
    if (!error) {
      write_mask = exec_mask;
    }
    return error;
  }
  // Validate before reading sources or committing any numeric/status output.
  if (int error =
          numeric<Op, 32>(numeric_flags, 0, modifiers, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  modifiers = rdna3_output_modifiers(numeric_flags, modifiers);
  const bool scaling = modifiers & RJ_ARITH_ALU_OMOD_MASK;
  uint32_t intermediate[32]{};
  uint32_t *output = intermediate;
  if (int error = numeric<Op, 32>(
          numeric_flags | (scaling ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0), exec_mask,
          modifiers & ~(RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP), &output, a, b, c)) {
    return error;
  }
  const uint32_t status = causes<Op>(numeric_flags, exec_mask, uint32_t(modifiers), a[0], b[0],
                                     Op == Operation::Fma ? c[0] : nullptr, intermediate);
  // Reuse the same arithmetic for the final OMOD/CLAMP, without a second model.
  if (int error = numeric<Op, 32>(numeric_flags, exec_mask, modifiers, d, a, b, c)) {
    return error;
  }
  *trapsts |= status;
  write_mask = exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Stage both halves before publishing a Wave64 result or accumulated status.
template <Operation Op>
int run64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
          uint32_t *trapsts) {
  const uint64_t numeric_flags =
      resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (!reporting(flags)) {
    return numeric<Op, 64>(numeric_flags, exec_mask, modifiers, d, a, b, c);
  }
  uint32_t half_write_mask = 0;
  if (int error = run32<Op>(flags, 0, modifiers, nullptr, nullptr, nullptr, nullptr, nullptr,
                            half_write_mask)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  uint32_t results[64]{}, status = 0;
  uint64_t write_mask = 0;
  for (int half = 0; half < 2; ++half) {
    const uint32_t half_exec_mask = uint32_t(exec_mask >> (half * 32));
    if (!half_exec_mask) {
      continue;
    }
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = nullptr;
    if constexpr (Op == Operation::Fma) {
      z = c[0] + half * 32;
    }
    uint32_t *out = results + half * 32;
    if (int error =
            run32<Op>(flags, half_exec_mask, dpp_half_wave_instruction_flags(modifiers, half), &out,
                      &x, &y, &z, &status, half_write_mask)) {
      return error;
    }
    write_mask |= uint64_t(half_write_mask) << (half * 32);
  }
  const uint32_t *source = results;
  const uint64_t copy_flags = flags & (RJ_ARITH_CPU_MASK | RJ_ARITH_CPU_FEATURES_MASK);
  if (int error = rj_arith_v_mov_b32_w64(copy_flags, write_mask, 0, d, &source)) {
    return error;
  }
  *trapsts |= status;
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_add_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<Operation::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts,
                               write_mask);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_add_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  return run64<Operation::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_sub_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<Operation::Sub>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts,
                               write_mask);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_sub_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  return run64<Operation::Sub>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_subrev_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<Operation::Subrev>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts,
                                  write_mask);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_subrev_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *trapsts) {
  return run64<Operation::Subrev>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_mul_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<Operation::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts,
                               write_mask);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_mul_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  return run64<Operation::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr, trapsts);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_fma_f32_w32_gfx1150(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c,
                                   uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<Operation::Fma>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts,
                               write_mask);
}

// Evaluate inherited gfx1150 arithmetic; Compatibility may accumulate legacy FP causes.
int rj_arith_v_fma_f32_w64_gfx1150(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c,
                                   uint32_t *trapsts) {
  return run64<Operation::Fma>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}
