// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cube-map face selection, coordinates, and major-axis magnitude.
// Owns public entry points and their validation/dispatch policy.

#include "graphics/cube.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

using rocjitsu::arith::Cube;

// Adapted from rocjitsu shared/cube.h.
// Magnitude used for cube comparisons, flushing subnormals to zero.
inline uint32_t cube_magnitude(uint32_t raw) {
  uint32_t magnitude = raw & 0x7fffffff;
  return magnitude < 0x00800000 ? 0 : magnitude;
}

// Directed-overflow rule borrowed from rocjitsu shared/cube.h.
inline uint32_t cube_overflow(uint32_t sign, unsigned round_mode) {
  const bool finite = round_mode == 3 || (round_mode == 1 && sign) || (round_mode == 2 && !sign);
  return sign | (finite ? 0x7f7fffffU : 0x7f800000U);
}

// Execute cube arithmetic with source/output modifiers and guest-directed
// overflow handling. Optionally quiet selected NaNs, retaining sign/payload.
// CLAMP maps NaNs to zero only when clamp_nan is set.
// Integer operations preserve host floating-point state.
template <Cube Op>
inline uint32_t cube_value(uint32_t x, uint32_t y, uint32_t z, uint32_t mode, bool quiet_nan,
                           bool clamp_nan, unsigned round_mode = 0) {
  x = (x & (mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : UINT32_MAX)) ^
      (mode & RJ_ARITH_ALU_NEG_A ? 0x80000000 : 0);
  y = (y & (mode & RJ_ARITH_ALU_ABS_B ? 0x7fffffff : UINT32_MAX)) ^
      (mode & RJ_ARITH_ALU_NEG_B ? 0x80000000 : 0);
  z = (z & (mode & RJ_ARITH_ALU_ABS_C ? 0x7fffffff : UINT32_MAX)) ^
      (mode & RJ_ARITH_ALU_NEG_C ? 0x80000000 : 0);
  uint32_t ax = cube_magnitude(x), ay = cube_magnitude(y), az = cube_magnitude(z);
  // Ties choose Z before Y before X; NaNs must follow the same axis-selection rules.
  bool z_axis = az <= 0x7f800000 && ax <= 0x7f800000 && ay <= 0x7f800000 && az >= ax && az >= ay;
  bool y_axis = ay <= 0x7f800000 && ax <= 0x7f800000 && ay >= ax;
  uint32_t major = z_axis ? z : y_axis ? y : x;
  uint32_t magnitude = cube_magnitude(major);
  bool negative = (major >> 31) && magnitude && magnitude <= 0x7f800000;
  uint32_t sign = negative ? 0x80000000 : 0, result;
  if constexpr (Op == Cube::Id) {
    result = z_axis   ? (negative ? 0x40a00000 : 0x40800000)
             : y_axis ? (negative ? 0x40400000 : 0x40000000)
                      : (negative ? 0x3f800000 : 0);
  } else if constexpr (Op == Cube::Sc) {
    result = z_axis ? x ^ sign : y_axis ? x : z ^ sign ^ 0x80000000;
  } else if constexpr (Op == Cube::Tc) {
    result = !z_axis && y_axis ? z ^ sign : y ^ 0x80000000;
  } else {
    result = magnitude >= 0x7f000000 ? cube_overflow(major & 0x80000000, round_mode)
                                     : major + 0x00800000;
    if (magnitude >= 0x7f800000) {
      result = major;
    }
    if (!magnitude) {
      result = round_mode == 2 ? major & 0x80000000 : 0;
    }
  }
  if (quiet_nan && (result & 0x7fffffff) > 0x7f800000) {
    result |= 0x00400000;
  }
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    uint32_t exponent = result & 0x7f800000, output_sign = result & 0x80000000;
    if (!exponent) {
      result = 0;
    } else if (exponent != 0x7f800000) {
      if (omod == 3) {
        result = exponent == 0x00800000 ? output_sign : result - 0x00800000;
      } else {
        result = exponent >= 0x7f800000 - omod * 0x00800000 ? cube_overflow(output_sign, round_mode)
                                                            : result + omod * 0x00800000;
      }
    }
  }
  if ((mode & RJ_ARITH_ALU_CLAMP) && (clamp_nan || (result & 0x7fffffff) <= 0x7f800000)) {
    if ((result >> 31) || result > 0x7f800000) {
      result = 0;
    } else if (result > 0x3f800000) {
      result = 0x3f800000;
    }
  }
  return result;
}

// Evaluate cube-map coordinates or face selection for the selected lanes. Reject invalid
// controls before changing outputs.
template <rocjitsu::arith::Cube Op, bool Rdna3 = false>
int execute(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return execute<Op, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  // Cube comparisons always flush denormals; only IEEE gates RDNA3 OMOD.
  if constexpr (Rdna3) {
    if (flags & RJ_ARITH_FP_IEEE_MODE) {
      mode &= ~RJ_ARITH_ALU_OMOD_MASK;
    }
  }
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~0x1ffU,
          RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const unsigned round_mode = (flags >> 24) & 3;
  const bool quiet_nan = !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE);
  const bool clamp_nan = !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::cube_x86_64_v4<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan,
                                        clamp_nan, round_mode);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::cube_x86_64_v3<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan,
                                        clamp_nan, round_mode);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] =
        cube_value<Op>(a[0][lane], b[0][lane], c[0][lane], mode, quiet_nan, clamp_nan, round_mode);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate cube-map coordinates or face selection for Wave64. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Cube Op, bool Rdna3 = false>
int execute_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = execute<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = execute<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)),
                                       half_instruction_flags, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Select the cube-map face for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubeid_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Id>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map S coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubesc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Sc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map T coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubetc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Tc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map major-axis magnitude for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubema_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Ma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the cube-map face for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubeid_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Id>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map S coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubesc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Sc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map T coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubetc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Tc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map major-axis magnitude for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubema_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Ma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the cube-map face for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubeid_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Id, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Select the cube-map face for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubeid_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Id, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, c);
}

// Compute the cube-map S coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubesc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Sc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map S coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubesc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Sc, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, c);
}

// Compute the cube-map T coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubetc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Tc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map T coordinate for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubetc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Tc, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, c);
}

// Compute the cube-map major-axis magnitude for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubema_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute<rocjitsu::arith::Cube::Ma, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute the cube-map major-axis magnitude for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cubema_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return execute_wave64<rocjitsu::arith::Cube::Ma, true>(flags, exec_mask, instruction_flags, d, a,
                                                         b, c);
}
