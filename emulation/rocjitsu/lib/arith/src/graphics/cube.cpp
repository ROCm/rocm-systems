// SPDX-License-Identifier: MIT

// Cube-map face selection, coordinates, and major-axis magnitude.
// Owns public entry points and their validation/dispatch policy.

#include "graphics/cube.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

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
      mode &= ~RJ_ARITH_ALU_OMOD_HALF;
    }
  }
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~0x1ffU, true,
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
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::cube_x86_64_v4<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan,
                                        clamp_nan, round_mode);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::cube_x86_64_v3<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan,
                                        clamp_nan, round_mode);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::cube_value<Op>(a[0][lane], b[0][lane], c[0][lane], mode,
                                                   quiet_nan, clamp_nan, round_mode);
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
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = execute<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                       &av, &bv, &cv)) {
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
