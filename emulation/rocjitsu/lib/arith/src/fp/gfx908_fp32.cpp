// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx908 ADD/MUL Compatibility retains the legacy IEEE arithmetic classifier,
// including raw input-denormal causes and explicit guest output flushing.
// Numerical results still use the shared guest scalar/SIMD arithmetic models.

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

// Call the existing ADD/MUL numerical model for the requested wave batch.
template <bool Multiply, int Width>
int numeric(uint64_t flags, uint64_t exec_mask, uint64_t modifiers, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if constexpr (Width == 32) {
    return (Multiply ? rj_arith_v_mul_f32_w32 : rj_arith_v_add_f32_w32)(
        flags, uint32_t(exec_mask), modifiers, d, a, b, nullptr);
  } else {
    return (Multiply ? rj_arith_v_mul_f32_w64 : rj_arith_v_add_f32_w64)(flags, exec_mask, modifiers,
                                                                        d, a, b, nullptr);
  }
}

// Classify one inherited IEEE operation, then apply the guest output-flush rule.
// Inputs are already modified and input-flushed; result is IEEE-rounded FP32.
uint32_t ieee_causes(uint64_t flags, uint32_t a, uint32_t b, uint32_t c, uint32_t &result) {
  uint32_t causes = compatibility_fma_causes<false>(flags, a, b, c, result);
  const uint32_t magnitude = result & 0x7fffffffU;
  if ((causes & RJ_ARITH_EXCEPTION_INEXACT) && magnitude == 0x00800000U) {
    // IEEE tininess precedes subnormal packing. Only a result rounded up to
    // minimum normal can disagree with the inherited packed-result test.
    const uint64_t exact = fma_finite_round_odd(a, b, c);
    const unsigned rounding = (flags & RJ_ARITH_FP_ROUND_MASK) >> 24;
    if (!(fma_narrow_f32(exact, /*flush_output=*/true, rounding) & 0x7fffffffU)) {
      causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
  }
  if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && magnitude && magnitude < 0x00800000U) {
    // The old explicit flush reports even an exactly representable subnormal.
    causes |= RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    result &= 0x80000000U;
  }
  return causes;
}

// Classify active modified inputs and IEEE intermediates before destination CLAMP.
template <bool Multiply>
uint32_t causes(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, const uint32_t *a,
                const uint32_t *b, const uint32_t *intermediate) {
  uint32_t status = 0;
  const unsigned omod = (modifiers >> 6) & 3;
  for (int lane = 0; lane < 32; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    const uint32_t raw_a = a[lane] & 0x7fffffffU, raw_b = b[lane] & 0x7fffffffU;
    if ((raw_a && raw_a < 0x00800000U) || (raw_b && raw_b < 0x00800000U)) {
      status |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    }
    const uint32_t x = alu_input_bits(a[lane], modifiers, flags);
    const uint32_t y = alu_input_bits(b[lane], modifiers >> 1, flags);
    uint32_t result = intermediate[lane];
    status |= ieee_causes(flags, x, Multiply ? y : 0x3f800000U,
                          Multiply ? (x ^ y) & 0x80000000U : y, result);
    if (omod) {
      const uint32_t scale = omod == 1 ? 0x40000000U : omod == 2 ? 0x40800000U : 0x3f000000U;
      const uint32_t zero = result & 0x80000000U;
      uint32_t scaled =
          fma_value_f32(result, scale, zero, flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, true);
      status |= ieee_causes(flags, result, scale, zero, scaled);
    }
  }
  return status;
}

// Evaluate one half-wave, returning its effective DPP store/status mask.
template <bool Multiply>
int run32(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts,
          uint32_t &write_mask) {
  write_mask = 0;
  if (modifiers >> 32) {
    return execute_dpp(flags, exec_mask, modifiers, a,
                       [&](uint32_t exec_mask, const uint32_t *const *source) {
                         return run32<Multiply>(flags, exec_mask, uint32_t(modifiers), d, source, b,
                                                trapsts, write_mask);
                       });
  }
  const uint64_t numeric_flags =
      resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if ((flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    const int error = numeric<Multiply, 32>(numeric_flags, exec_mask, modifiers, d, a, b);
    if (!error) {
      write_mask = exec_mask;
    }
    return error;
  }
  // Reject controls before reading operands or committing either output.
  if (int error = numeric<Multiply, 32>(numeric_flags, 0, modifiers, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  modifiers = rdna3_output_modifiers(numeric_flags, modifiers);
  uint32_t intermediate[32]{};
  uint32_t *output = intermediate;
  // Legacy status used IEEE arithmetic before its separate explicit guest flush.
  // Keep that intermediate distinct from the final guest numerical result.
  if (int error = numeric<Multiply, 32>(
          numeric_flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, exec_mask,
          modifiers & ~(RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP), &output, a, b)) {
    return error;
  }
  const uint32_t status =
      causes<Multiply>(numeric_flags, exec_mask, uint32_t(modifiers), a[0], b[0], intermediate);
  if (int error = numeric<Multiply, 32>(numeric_flags, exec_mask, modifiers, d, a, b)) {
    return error;
  }
  *trapsts |= status;
  write_mask = exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Stage both Wave64 halves before publishing aliased results and accumulated causes.
template <bool Multiply>
int run64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  const uint64_t numeric_flags =
      resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if ((flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    return numeric<Multiply, 64>(numeric_flags, exec_mask, modifiers, d, a, b);
  }
  uint32_t half_write_mask = 0;
  if (int error = run32<Multiply>(flags, 0, modifiers, nullptr, nullptr, nullptr, nullptr,
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
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    uint32_t *out = results + half * 32;
    if (int error =
            run32<Multiply>(flags, half_exec_mask, dpp_half_wave_instruction_flags(modifiers, half),
                            &out, &x, &y, &status, half_write_mask)) {
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

// Evaluate the inherited gfx908 numeric and Compatibility status policies.
int rj_arith_v_add_f32_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<false>(flags, exec_mask, instruction_flags, d, a, b, trapsts, write_mask);
}

// Evaluate the inherited gfx908 numeric and Compatibility status policies.
int rj_arith_v_add_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  return run64<false>(flags, exec_mask, instruction_flags, d, a, b, trapsts);
}

// Evaluate the inherited gfx908 numeric and Compatibility status policies.
int rj_arith_v_mul_f32_w32_gfx908(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  uint32_t write_mask = 0;
  return run32<true>(flags, exec_mask, instruction_flags, d, a, b, trapsts, write_mask);
}

// Evaluate the inherited gfx908 numeric and Compatibility status policies.
int rj_arith_v_mul_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  return run64<true>(flags, exec_mask, instruction_flags, d, a, b, trapsts);
}
