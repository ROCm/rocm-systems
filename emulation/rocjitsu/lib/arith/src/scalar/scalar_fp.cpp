// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar floating-point arithmetic and selection, with optional FP status.
// Owns public entry points and their validation/dispatch policy.

// Scalar operations follow rocjitsu generated/shared/execute_shared.h. GFX1201
// captures establish the output-flush exceptions and multiplication threshold.
// ADD/SUB/MUL with directed rounding or exact nonfinite inputs use the VALU FMA
// models; selection shares the VALU minmax model. GFX1201 captures match every
// form in every MODE setting.

#include "common/float_encoding.h"
#include "common/validation.h"
#include "compare/float_compare_impl.h"
#include "fp/binary.h"
#include "fp/fma_scalar.h"
#include "fp/half_fma_scalar.h"
#include "fp/minmax_impl.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/fp_status.h"
#include "scalar/gfx1150_policy.h"

#include <stdint.h>

namespace {

// rocjitsu lowers ADD/SUB to FMA(A, 1, +/-B) and MUL to FMA(A, B, a zero with
// the product sign). SUB negates B before NaN selection, so a NaN B changes sign.
template <rocjitsu::arith::Binary Op, bool Half, bool Gfx1150 = false>
uint32_t fused_value(uint64_t flags, uint32_t a, uint32_t b) {
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, one = Half ? 0x3c00 : 0x3f800000;
  uint32_t c;
  if constexpr (Op == rocjitsu::arith::Binary::Mul) {
    c = (a ^ b) & sign;
  } else {
    c = Op == rocjitsu::arith::Binary::Sub ? b ^ sign : b;
    b = one;
  }
  if constexpr (Half) {
    return rocjitsu::arith::half_fma_value<Gfx1150>(uint16_t(a), uint16_t(b), uint16_t(c), 0,
                                                    flags);
  } else {
    return rocjitsu::arith::fma_value_f32(a, b, c, flags, Gfx1150);
  }
}

// Evaluate scalar FP arithmetic. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Binary Op, bool Half, bool Gfx1150 = false>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b) {
  using rocjitsu::arith::Binary;
  static_assert(Op == Binary::Add || Op == Binary::Sub || Op == Binary::Mul ||
                Op == Binary::MinNum || Op == Binary::MaxNum || Op == Binary::Minimum ||
                Op == Binary::Maximum);
  constexpr bool arithmetic = Op == Binary::Add || Op == Binary::Sub || Op == Binary::Mul;
  if (int error = rocjitsu::arith::validate(
          flags, mode,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              (arithmetic ? RJ_ARITH_FP_ROUND_MASK : 0) | (Gfx1150 ? RJ_ARITH_FP_IEEE_MODE : 0))) {
    return error;
  }
  constexpr uint32_t inf = Half ? 0x7c00 : 0x7f800000, sign = Half ? 0x8000 : 0x80000000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
  }
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf)) {
      a &= sign;
    }
    if (!(b & inf)) {
      b &= sign;
    }
  }
  if constexpr (arithmetic) {
    // As for VALU exact ADD/MUL, nearest-even arithmetic on finite inputs equals the
    // FMA model natively, given default host FP state; gfx1201 captures agree.
    const bool finite = (a & inf) != inf && (b & inf) != inf;
    if ((flags & RJ_ARITH_FP_ROUND_MASK) || (rocjitsu::arith::is_exact(flags) && !finite)) {
      *d = fused_value<Op, Half, Gfx1150>(flags, a, b);
      return RJ_ARITH_SUCCESS;
    }
  }
  float x = Half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
  float y = Half ? rocjitsu::arith::f16_to_float(uint16_t(b)) : rocjitsu::arith::as_float(b), value;
  if constexpr (Op == rocjitsu::arith::Binary::Add) {
    value = x + y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::Sub) {
    value = x - y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::Mul) {
    value = x * y;
  }
  if constexpr (Op == rocjitsu::arith::Binary::MinNum) {
    if constexpr (Gfx1150) {
      value = rocjitsu::arith::minmax_legacy<false>(x, y, flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value = rocjitsu::arith::minmax<false, false>(x, y);
    }
  }
  if constexpr (Op == rocjitsu::arith::Binary::MaxNum) {
    if constexpr (Gfx1150) {
      value = rocjitsu::arith::minmax_legacy<true>(x, y, flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value = rocjitsu::arith::minmax<true, false>(x, y);
    }
  }
  if constexpr (Op == rocjitsu::arith::Binary::Minimum) {
    value = rocjitsu::arith::minmax<false, true>(x, y);
  }
  if constexpr (Op == rocjitsu::arith::Binary::Maximum) {
    value = rocjitsu::arith::minmax<true, true>(x, y);
  }
  uint32_t result = Half ? rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL)
                         : rocjitsu::arith::as_bits(value);
  if constexpr (Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Sub ||
                Op == rocjitsu::arith::Binary::Mul) {
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      if (!(result & inf)) {
        result &= sign;
      }
      if constexpr (Op == rocjitsu::arith::Binary::Mul) {
        // GPU tininess detection precedes destination subnormal rounding. The
        // threshold is a quarter subnormal ULP below the smallest normal; ties
        // round to the normal value. Double holds the exact FP32 product.
        constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
        if (__builtin_fabs(double(x) * double(y)) < threshold) {
          result &= sign;
        }
      }
    }
  }
  *d = result;
  return RJ_ARITH_SUCCESS;
}

// Preserve gfx12 Compatibility status around the shared numerical operation.
// Explicit reporting modes retain their existing qualification restrictions.
template <rocjitsu::arith::Binary Op, bool Half, bool Gfx1150 = false>
int run_policy(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b,
               uint32_t *status) {
  using namespace rocjitsu::arith;
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if constexpr (Gfx1150) {
    if (int error = prepare_gfx1150_scalar(flags)) {
      return error;
    }
  }
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = validate_semantics(flags)) {
    return error;
  }
  const bool report = reports_fp_status(flags);
  if constexpr (Gfx1150) {
    if (report) {
      return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
    }
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint64_t numeric_flags = without_fp_status(flags);
  int error = run<Op, Half, Gfx1150>(numeric_flags, uint32_t(mode), d, a, b);
  if (error || (!compatibility && !report)) {
    return error;
  }
  if constexpr (Op == Binary::Add || Op == Binary::Sub || Op == Binary::Mul) {
    constexpr uint32_t sign = Half ? 0x8000 : 0x80000000;
    constexpr uint32_t one = Half ? 0x3c00 : 0x3f800000;
    const uint32_t c = Op == Binary::Mul ? (a ^ b) & sign : Op == Binary::Sub ? b ^ sign : b;
    *status |=
        scalar_fma_causes<Half, Gfx1150>(numeric_flags, a, Op == Binary::Mul ? b : one, c, *d);
  } else {
    *status |= float_compare_exceptions<Half ? 16 : 32>(
        a, b, numeric_flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  return RJ_ARITH_SUCCESS;
}

// Keep legacy gfx1250 scalar arithmetic gradual, while selection honors input MODE.
// No FP status model is inherited for these instructions.
template <rocjitsu::arith::Binary Op, bool Half>
int run_gfx1250(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate(
          flags, mode, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (compatibility) {
    flags &= ~RJ_ARITH_FP16_OVFL;
    if constexpr (Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Sub ||
                  Op == rocjitsu::arith::Binary::Mul) {
      flags &= ~(RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
    }
  }
  return run<Op, Half>(flags, 0, d, a, b);
}

} // namespace

// Add the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_add_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Add, false>(flags, instruction_flags, d, a, b,
                                                         excp_flag_user);
}

// Add the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_add_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Add, true>(flags, instruction_flags, d, a, b,
                                                        excp_flag_user);
}

// Subtract B from A.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_sub_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Sub, false>(flags, instruction_flags, d, a, b,
                                                         excp_flag_user);
}

// Subtract B from A.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_sub_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Sub, true>(flags, instruction_flags, d, a, b,
                                                        excp_flag_user);
}

// Multiply the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_mul_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Mul, false>(flags, instruction_flags, d, a, b,
                                                         excp_flag_user);
}

// Multiply the operands.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_mul_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Mul, true>(flags, instruction_flags, d, a, b,
                                                        excp_flag_user);
}

// Select the minimum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_min_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::MinNum, false>(flags, instruction_flags, d, a, b,
                                                            excp_flag_user);
}

// Select the minimum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_min_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::MinNum, true>(flags, instruction_flags, d, a, b,
                                                           excp_flag_user);
}

// Select the maximum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_max_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::MaxNum, false>(flags, instruction_flags, d, a, b,
                                                            excp_flag_user);
}

// Select the maximum with numeric-preferred NaN selection.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_max_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::MaxNum, true>(flags, instruction_flags, d, a, b,
                                                           excp_flag_user);
}

// Select the NaN-propagating minimum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_minimum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Minimum, false>(flags, instruction_flags, d, a, b,
                                                             excp_flag_user);
}

// Select the NaN-propagating minimum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_minimum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Minimum, true>(flags, instruction_flags, d, a, b,
                                                            excp_flag_user);
}

// Select the NaN-propagating maximum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_maximum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Maximum, false>(flags, instruction_flags, d, a, b,
                                                             excp_flag_user);
}

// Select the NaN-propagating maximum.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_maximum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<rocjitsu::arith::Binary::Maximum, true>(flags, instruction_flags, d, a, b,
                                                            excp_flag_user);
}

// Evaluate gfx1250 scalar add; strict empirical modes are unsupported.
int rj_arith_s_add_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Add, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar add; strict empirical modes are unsupported.
int rj_arith_s_add_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Add, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar sub; strict empirical modes are unsupported.
int rj_arith_s_sub_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Sub, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar sub; strict empirical modes are unsupported.
int rj_arith_s_sub_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Sub, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar mul; strict empirical modes are unsupported.
int rj_arith_s_mul_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Mul, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar mul; strict empirical modes are unsupported.
int rj_arith_s_mul_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Mul, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar min_num; strict empirical modes are unsupported.
int rj_arith_s_min_num_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::MinNum, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar min_num; strict empirical modes are unsupported.
int rj_arith_s_min_num_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::MinNum, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar max_num; strict empirical modes are unsupported.
int rj_arith_s_max_num_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::MaxNum, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar max_num; strict empirical modes are unsupported.
int rj_arith_s_max_num_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::MaxNum, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar minimum; strict empirical modes are unsupported.
int rj_arith_s_minimum_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Minimum, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar minimum; strict empirical modes are unsupported.
int rj_arith_s_minimum_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Minimum, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar maximum; strict empirical modes are unsupported.
int rj_arith_s_maximum_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Maximum, true>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1250 scalar maximum; strict empirical modes are unsupported.
int rj_arith_s_maximum_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t b,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<rocjitsu::arith::Binary::Maximum, false>(flags, instruction_flags, d, a, b);
}

// Evaluate gfx1150 scalar add, preserving the inherited IEEE/NaN policy.
int rj_arith_s_add_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Add, true, true>(flags, instruction_flags, d, a, b,
                                                              fp_status);
}

// Evaluate gfx1150 scalar sub, preserving the inherited IEEE/NaN policy.
int rj_arith_s_sub_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Sub, true, true>(flags, instruction_flags, d, a, b,
                                                              fp_status);
}

// Evaluate gfx1150 scalar mul, preserving the inherited IEEE/NaN policy.
int rj_arith_s_mul_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Mul, true, true>(flags, instruction_flags, d, a, b,
                                                              fp_status);
}

// Evaluate gfx1150 scalar min, preserving the inherited IEEE/NaN policy.
int rj_arith_s_min_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::MinNum, true, true>(flags, instruction_flags, d, a, b,
                                                                 fp_status);
}

// Evaluate gfx1150 scalar max, preserving the inherited IEEE/NaN policy.
int rj_arith_s_max_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::MaxNum, true, true>(flags, instruction_flags, d, a, b,
                                                                 fp_status);
}

// Evaluate gfx1150 scalar add, preserving the inherited IEEE/NaN policy.
int rj_arith_s_add_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Add, false, true>(flags, instruction_flags, d, a, b,
                                                               fp_status);
}

// Evaluate gfx1150 scalar sub, preserving the inherited IEEE/NaN policy.
int rj_arith_s_sub_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Sub, false, true>(flags, instruction_flags, d, a, b,
                                                               fp_status);
}

// Evaluate gfx1150 scalar mul, preserving the inherited IEEE/NaN policy.
int rj_arith_s_mul_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::Mul, false, true>(flags, instruction_flags, d, a, b,
                                                               fp_status);
}

// Evaluate gfx1150 scalar min, preserving the inherited IEEE/NaN policy.
int rj_arith_s_min_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::MinNum, false, true>(flags, instruction_flags, d, a, b,
                                                                  fp_status);
}

// Evaluate gfx1150 scalar max, preserving the inherited IEEE/NaN policy.
int rj_arith_s_max_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b, uint32_t *fp_status) {
  return run_policy<rocjitsu::arith::Binary::MaxNum, false, true>(flags, instruction_flags, d, a, b,
                                                                  fp_status);
}
