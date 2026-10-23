// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 interpolation stages with source-lane selection.
// Owns public entry points and their validation/dispatch policy.

// Quad broadcasts follow both RDNA3 and RDNA4 ISA guides, section 12.3.

#include "common/alu.h"
#include "common/internal.h"
#include "graphics/interp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Evaluate an FP32 interpolation stage for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool P2>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_INTERP_WAIT_EXP_MASK;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::interp32_x86_64_v4<P2>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::interp32_x86_64_v3<P2>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    unsigned quad = lane & ~3u;
    float x = rocjitsu::arith::alu_input(a[0][quad + (P2 ? 2 : 1)], mode);
    float y = rocjitsu::arith::alu_input(b[0][lane], mode >> 1);
    float z = rocjitsu::arith::alu_input(c[0][P2 ? lane : quad], mode >> 2);
    result[lane] = rocjitsu::arith::as_bits(rocjitsu::arith::alu_output(std::fma(x, y, z), mode));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate an FP32 interpolation stage for Wave64. Reject invalid controls before changing
// outputs.
template <bool P2>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<P2>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }

  // Quad broadcasts never cross a 32-lane half.
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error =
            run<P2>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Execute INTERP_P10_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P10_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
