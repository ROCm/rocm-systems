// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 add/subtract/multiply, including packed forms and guest FP status and min/max selection.
// Owns public entry points and their validation/dispatch policy.

#include "fp/half_binary.h"
#include "common/alu.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "common/packed_alu.h"
#include "fp/binary.h"
#include "fp/binary_impl.h"
#include "fp/float_output.h"
#include "fp/half_fma_scalar.h"
#include "fp/minmax_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Guest-mode packed ADD/MUL follows rocjitsu fp_mode::packed_binary_f16.
// Finite FP16 addition and multiplication are exact in FP64; narrow with
// integer rounding rather than changing the CPU environment.
template <rocjitsu::arith::Binary Op, bool Rdna3>
int packed_arithmetic_modes(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  static_assert(Op == rocjitsu::arith::Binary::Add || Op == rocjitsu::arith::Binary::Mul);
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int half = 0; half < 2; ++half) {
      const uint32_t m =
          rocjitsu::arith::packed_half_instruction_flags(instruction_flags, half != 0);
      uint16_t x = uint16_t(a[0][lane] >> (m & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      uint16_t y = uint16_t(b[0][lane] >> (m & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
      if (m & RJ_ARITH_ALU_NEG_A) {
        x ^= 0x8000;
      }
      if (m & RJ_ARITH_ALU_NEG_B) {
        y ^= 0x8000;
      }
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        if ((x & 0x7fff) < 0x400) {
          x &= 0x8000;
        }
        if ((y & 0x7fff) < 0x400) {
          y &= 0x8000;
        }
      }
      const uint16_t mx = x & 0x7fff, my = y & 0x7fff;
      uint16_t encoded;
      if (mx > 0x7c00 || my > 0x7c00) {
        encoded = (mx > 0x7c00 ? x : y) | ((!Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE)) ? 0x200 : 0);
      } else if constexpr (Op == rocjitsu::arith::Binary::Add) {
        if (mx == 0x7c00 && my == 0x7c00 && ((x ^ y) & 0x8000)) {
          encoded = 0xfe00;
        } else if (mx == 0x7c00 || my == 0x7c00) {
          encoded = mx == 0x7c00 ? x : y;
        } else {
          double value =
              double(rocjitsu::arith::f16_to_float(x)) + double(rocjitsu::arith::f16_to_float(y));
          if (value == 0 && ((x ^ y) & 0x8000)) {
            value = rounding == 2 ? -0.0 : 0.0;
          }
          encoded = rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(value),
                                                     flags & RJ_ARITH_FP16_OVFL, rounding);
        }
      } else {
        if ((mx == 0 && my == 0x7c00) || (my == 0 && mx == 0x7c00)) {
          encoded = 0xfe00;
        } else if (mx == 0x7c00 || my == 0x7c00) {
          encoded = ((x ^ y) & 0x8000) | 0x7c00;
        } else {
          encoded = rocjitsu::arith::half_fma_narrow(
              rocjitsu::arith::bit_cast<uint64_t>(double(rocjitsu::arith::f16_to_float(x)) *
                                                  double(rocjitsu::arith::f16_to_float(y))),
              flags & RJ_ARITH_FP16_OVFL, rounding);
        }
      }
      const bool preserve_nan =
          Rdna3 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (encoded & 0x7fff) > 0x7c00;
      if ((m & RJ_ARITH_ALU_CLAMP) && !preserve_nan) {
        encoded = rocjitsu::arith::half_fma_clamp(encoded);
      }

      // Unlike FMA, packed binary operations test the packed result's exponent.
      if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (encoded & 0x7fff) < 0x400) {
        encoded &= 0x8000;
      }
      result[lane] |= uint32_t(encoded) << (half * 16);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Ordinary directed ADD/SUB/MUL follows rocjitsu fp_mode::arithmetic_f16:
// finite FP16 operands produce an exact FP64 intermediate, then integer
// narrowing applies the guest rounding mode without changing host controls.
template <rocjitsu::arith::Binary Op, bool Rdna3>
int arithmetic_modes(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     bool compatibility) {
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const unsigned a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const unsigned b_shift = instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  const unsigned d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const unsigned omod = (instruction_flags >> 6) & 3;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t x = uint16_t(a[0][lane] >> a_shift);
    uint16_t y = uint16_t(b[0][lane] >> b_shift);
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      if ((x & 0x7fff) < 0x400) {
        x &= 0x8000;
      }
      if ((y & 0x7fff) < 0x400) {
        y &= 0x8000;
      }
    }
    if (instruction_flags & RJ_ARITH_ALU_ABS_A) {
      x &= 0x7fff;
    }
    if (instruction_flags & RJ_ARITH_ALU_ABS_B) {
      y &= 0x7fff;
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_A) {
      x ^= 0x8000;
    }
    if (instruction_flags & RJ_ARITH_ALU_NEG_B) {
      y ^= 0x8000;
    }
    if constexpr (Op == rocjitsu::arith::Binary::Sub) {
      y ^= 0x8000;
    }
    if constexpr (Op == rocjitsu::arith::Binary::Subrev) {
      x ^= 0x8000;
    }
    const double lhs = rocjitsu::arith::f16_to_float(x), rhs = rocjitsu::arith::f16_to_float(y);
    double value;
    if constexpr (Op == rocjitsu::arith::Binary::Mul) {
      value = lhs * rhs;
    } else {
      value = lhs + rhs;
      if (value == 0 && ((x ^ y) & 0x8000)) {
        value = rounding == 2 ? -0.0 : 0.0;
      }
    }
    if (compatibility) {
      if (__builtin_isnan(lhs)) {
        value = lhs;
      } else if (__builtin_isnan(rhs)) {
        value = rhs;
      }
    }
    // Compatibility retains rocjitsu's FP16 rounding before OMOD. Loose keeps
    // the wider result so that scaling and destination rounding can be combined.
    if (compatibility && omod && __builtin_isfinite(value)) {
      value = rocjitsu::arith::f16_to_float(rocjitsu::arith::half_fma_narrow(
          rocjitsu::arith::bit_cast<uint64_t>(value), flags & RJ_ARITH_FP16_OVFL, rounding));
    }
    if (omod) {
      const double magnitude = value < 0 ? -value : value;
      if (magnitude < 0x1p-14) {
        value = 0;
      } else if (omod == 3 && magnitude < 0x1p-13) {
        value = value < 0 ? -0.0 : 0.0;
      }
      const double scales[] = {1, 2, 4, 0.5};
      value *= scales[omod];
    }
    if (compatibility && __builtin_isnan(value)) {
      value = rocjitsu::arith::bit_cast<double>(rocjitsu::arith::bit_cast<uint64_t>(value) |
                                                0x0008000000000000ULL);
    }
    const bool nan = value != value;
    if ((instruction_flags & RJ_ARITH_ALU_CLAMP) &&
        !(nan && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    }

    // The finite narrowing helper excludes NaNs/infinities.
    const uint64_t magnitude = rocjitsu::arith::double_bits(value) & 0x7fffffffffffffffULL;
    uint16_t encoded =
        magnitude >= 0x7ff0000000000000ULL
            ? rocjitsu::arith::float_to_f16(float(value), flags & RJ_ARITH_FP16_OVFL)
            : rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(value),
                                               flags & RJ_ARITH_FP16_OVFL, rounding);
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (encoded & 0x7fff) < 0x400) {
      encoded &= 0x8000;
    }
    result[lane] = encoded;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Computes masked scalar results; GuestDenormals enables guest input/output flushing.
template <rocjitsu::arith::Binary Op, bool Packed, bool GuestDenormals, bool Rdna3>
[[gnu::noinline]] int half_binary_scalar(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         bool compatibility, bool preserve_input_snan) {
  constexpr bool arithmetic = rocjitsu::arith::is_arithmetic(Op);
  // Each packed result half has independent source-half selectors and sign modifiers.
  uint32_t modes[] = {Packed
                          ? rocjitsu::arith::packed_half_instruction_flags(instruction_flags, false)
                          : uint32_t(instruction_flags),
                      rocjitsu::arith::packed_half_instruction_flags(instruction_flags, true)};
  int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      uint32_t m = modes[half];
      int a_shift = m & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
      int b_shift = m & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
      uint16_t a_bits = uint16_t(a[0][lane] >> a_shift);
      uint16_t b_bits = uint16_t(b[0][lane] >> b_shift);
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
        if ((a_bits & 0x7fff) < 0x0400) {
          a_bits &= 0x8000;
        }
        if ((b_bits & 0x7fff) < 0x0400) {
          b_bits &= 0x8000;
        }
      }
      float x = rocjitsu::arith::alu_input(
          rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(a_bits)), m);
      float y = rocjitsu::arith::alu_input(
          rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(b_bits)), m >> 1);
      float value;
      if constexpr (Op == rocjitsu::arith::Binary::Add) {
        value = x + y;
      }
      if constexpr (Op == rocjitsu::arith::Binary::Sub) {
        y = rocjitsu::arith::as_float(rocjitsu::arith::as_bits(y) ^ 0x80000000U);
        value = x + y;
      }
      if constexpr (Op == rocjitsu::arith::Binary::Subrev) {
        x = rocjitsu::arith::as_float(rocjitsu::arith::as_bits(x) ^ 0x80000000U);
        value = x + y;
      }
      if constexpr (Op == rocjitsu::arith::Binary::Mul) {
        value = x * y;
      }
      if constexpr (Op == rocjitsu::arith::Binary::MinLegacy ||
                    Op == rocjitsu::arith::Binary::MaxLegacy) {
        value = rocjitsu::arith::minmax_rdna3<Op == rocjitsu::arith::Binary::MaxLegacy>(
            x, y, flags & RJ_ARITH_FP_IEEE_MODE);
      }
      if constexpr (Op == rocjitsu::arith::Binary::MinNum) {
        value = rocjitsu::arith::minmax<false, false>(x, y);
      }
      if constexpr (Op == rocjitsu::arith::Binary::MaxNum) {
        value = rocjitsu::arith::minmax<true, false>(x, y);
      }
      if constexpr (Op == rocjitsu::arith::Binary::Minimum) {
        value = rocjitsu::arith::minmax<false, true>(x, y);
      }
      if constexpr (Op == rocjitsu::arith::Binary::Maximum) {
        value = rocjitsu::arith::minmax<true, true>(x, y);
      }

      // Host subtraction may keep a NaN subtrahend's old sign, and host ADD
      // need not preserve the guest's first-source NaN priority.
      if (arithmetic && compatibility) {
        if (__builtin_isnan(x)) {
          value = x;
        } else if (__builtin_isnan(y)) {
          value = y;
        }
      }
      if (arithmetic && compatibility && !Packed && (m & RJ_ARITH_ALU_OMOD_MASK)) {
        value = rocjitsu::arith::f16_to_float(
            rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL));
      }
      if (arithmetic && compatibility && __builtin_isnan(value)) {
        value = rocjitsu::arith::as_float(rocjitsu::arith::as_bits(value) | 0x00400000U);
      }
      if ((flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
          (rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000) {
        m &= ~RJ_ARITH_ALU_CLAMP;
      }
      uint16_t encoded;
      if constexpr (arithmetic) {
        encoded = rocjitsu::arith::float_to_f16(rocjitsu::arith::alu_output_f16(value, m),
                                                flags & RJ_ARITH_FP16_OVFL);
      } else {
        encoded = uint16_t(rocjitsu::arith::selection_output<16>(
            rocjitsu::arith::float_to_f16(value, false), m, flags));
      }
      if constexpr (Packed && Rdna3 && arithmetic) {
        // Packed gfx11 preserves input sNaNs when IEEE is clear. FP32/FP16
        // conversions must not accidentally impose the ordinary FP16 quieting rule.
        if (preserve_input_snan && !(m & RJ_ARITH_ALU_CLAMP)) {
          if ((a_bits & 0x7fff) > 0x7c00) {
            encoded = a_bits ^ (m & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0);
          } else if ((b_bits & 0x7fff) > 0x7c00) {
            encoded = b_bits ^ (m & RJ_ARITH_ALU_NEG_B ? 0x8000 : 0);
          }
        }
      }
      if (GuestDenormals && arithmetic && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
          (encoded & 0x7fff) < 0x0400) {
        encoded &= 0x8000;
      }
      result[lane] |= uint32_t(encoded) << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      if constexpr (Packed) {
        d[0][lane] = result[lane];
      } else {
        d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (result[lane] << d_shift);
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 binary arithmetic or selection for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::Binary Op, bool Packed = false, bool Rdna3 = false>
int half_binary(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, bool compatibility = false) {
  if (instruction_flags >> 32) {
    if constexpr (!Packed) {
      return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                          [&](uint32_t exec_mask, const uint32_t *const *source) {
                                            return half_binary<Op, Packed, Rdna3>(
                                                flags, exec_mask, uint32_t(instruction_flags), d,
                                                source, b, compatibility);
                                          });
    } else {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known =
      Packed ? RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B | RJ_ARITH_PK_NEG_HI_A |
                   RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_CLAMP | RJ_ARITH_PK_LO_A_HIGH |
                   RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_HI_A_LOW | RJ_ARITH_PK_HI_B_LOW
             : RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B |
                   RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A |
                   RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  constexpr bool arithmetic = rocjitsu::arith::is_arithmetic(Op);
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE;

  constexpr uint64_t mode_flags = RJ_ARITH_FP_ROUND_MASK | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error =
          rocjitsu::arith::validate(flags, instruction_flags & ~known, fp_flags | mode_flags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if constexpr (Rdna3 && !Packed) {
    instruction_flags = rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags);
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (!Packed && arithmetic) {
    if (flags & RJ_ARITH_FP_ROUND_MASK) {
      return arithmetic_modes<Op, Rdna3>(flags, exec_mask, uint32_t(instruction_flags), d, a, b,
                                         compatibility);
    }
  }
  if constexpr (Packed && arithmetic) {
    if ((flags & RJ_ARITH_FP_ROUND_MASK) ||
        ((instruction_flags & RJ_ARITH_PK_CLAMP) && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
      return packed_arithmetic_modes<Op, Rdna3>(flags, exec_mask, uint32_t(instruction_flags), d, a,
                                                b);
    }
  }
  const bool preserve_input_snan =
      Packed && Rdna3 && compatibility && !(flags & RJ_ARITH_FP_IEEE_MODE);
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_binary_x86_64_v3<Op, Packed>(
        flags, exec_mask, instruction_flags, d[0], a[0], b[0], compatibility, preserve_input_snan);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if constexpr (Packed) {
    if (!(flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      return half_binary_scalar<Op, Packed, false, Rdna3>(flags, exec_mask, instruction_flags, d, a,
                                                          b, compatibility, preserve_input_snan);
    }
  }
  return half_binary_scalar<Op, Packed, true, Rdna3>(flags, exec_mask, instruction_flags, d, a, b,
                                                     compatibility, preserve_input_snan);
}

// Evaluate FP16 binary arithmetic or selection for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Binary Op, bool Packed = false, bool Rdna3 = false>
int half_binary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       bool compatibility = false) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = half_binary<Op, Packed, Rdna3>(flags, 0, instruction_flags, nullptr, nullptr,
                                                 nullptr, compatibility)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error =
            half_binary<Op, Packed, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)),
                                           half_instruction_flags, &dest, &x, &y, compatibility)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Add, false, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, compatibility);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Add>(flags, exec_mask, instruction_flags, d, a, b,
                                                   compatibility);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Sub, false, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, compatibility);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Sub>(flags, exec_mask, instruction_flags, d, a, b,
                                                   compatibility);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Subrev, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Subrev>(flags, exec_mask, instruction_flags, d, a, b,
                                                      compatibility);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Mul, false, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, compatibility);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Mul>(flags, exec_mask, instruction_flags, d, a, b,
                                                   compatibility);
}

// Select the minimum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MinNum>(flags, exec_mask, instruction_flags, d, a, b);
}

// Select the maximum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MaxNum>(flags, exec_mask, instruction_flags, d, a, b);
}

// Select the NaN-propagating minimum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Minimum>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

// Select the NaN-propagating maximum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Maximum>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_add_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::Add, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, compatibility);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_mul_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::Mul, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, compatibility);
}

// Select the minimum with packed elements, numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_min_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::MinNum, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Select the maximum with packed elements, numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_max_num_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::MaxNum, true>(flags, exec_mask, instruction_flags, d,
                                                            a, b);
}

// Select the NaN-propagating minimum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_minimum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::Minimum, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Select the NaN-propagating maximum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_maximum_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary<rocjitsu::arith::Binary::Maximum, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MinLegacy, false, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MaxLegacy, false, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_min_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MinLegacy, true, true>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_max_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::MaxLegacy, true, true>(flags, exec_mask,
                                                                     instruction_flags, d, a, b);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Add, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Add the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_add_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Add>(flags, exec_mask, instruction_flags, d, a,
                                                          b, compatibility);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Sub, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Subtract B from A for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sub_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Sub>(flags, exec_mask, instruction_flags, d, a,
                                                          b, compatibility);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Subrev, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Subtract A from B for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_subrev_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Subrev>(flags, exec_mask, instruction_flags, d,
                                                             a, b, compatibility);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Mul, false, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mul_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Mul>(flags, exec_mask, instruction_flags, d, a,
                                                          b, compatibility);
}

// Select the minimum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MinNum>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Select the maximum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MaxNum>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Select the NaN-propagating minimum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_minimum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Minimum>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Select the NaN-propagating maximum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_maximum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Maximum>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_add_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Add, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, compatibility);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_mul_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Mul, true>(flags, exec_mask, instruction_flags,
                                                                d, a, b, compatibility);
}

// Select the minimum with packed elements, numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_min_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MinNum, true>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the maximum with packed elements, numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_max_num_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MaxNum, true>(flags, exec_mask,
                                                                   instruction_flags, d, a, b);
}

// Select the NaN-propagating minimum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_minimum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Minimum, true>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Select the NaN-propagating maximum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_maximum_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Maximum, true>(flags, exec_mask,
                                                                    instruction_flags, d, a, b);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_min_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MinLegacy, false, true>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_max_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MaxLegacy, false, true>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the minimum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_min_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MinLegacy, true, true>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Select the maximum with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_max_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::MaxLegacy, true, true>(
      flags, exec_mask, instruction_flags, d, a, b);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_add_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Add, true, true>(flags, exec_mask, instruction_flags,
                                                               d, a, b, compatibility);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_mul_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary<rocjitsu::arith::Binary::Mul, true, true>(flags, exec_mask, instruction_flags,
                                                               d, a, b, compatibility);
}

// Add the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_add_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Add, true, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}

// Multiply the operands with packed elements for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pk_mul_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return half_binary_wave64<rocjitsu::arith::Binary::Mul, true, true>(
      flags, exec_mask, instruction_flags, d, a, b, compatibility);
}
