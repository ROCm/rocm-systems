// SPDX-License-Identifier: MIT

// Whole-word, halfword, and relative-register swaps.
// Owns public entry points and their validation/dispatch policy.

#include "common/internal.h"
#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Swap VGPR values for the selected lanes. Reject invalid controls before changing outputs.
template <int Lanes>
int swap(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
         uint32_t *const *a) {
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t old_d = d[0][lane], old_a = a[0][lane];
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = old_a;
      a[0][lane] = old_d;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Both operands are destinations: an out-of-range index discards the entire
// instruction. Keep widened sums so a large base cannot wrap into the file.
template <int Lanes>
int swap_relative(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                  uint32_t *const *a, uint32_t d_count, uint32_t a_count, uint32_t d_base,
                  uint32_t a_base, uint32_t m0) {
  // Reject flags before forming pointers into either register-file view.
  if (int error = swap<Lanes>(flags, 0, mode, nullptr, nullptr)) {
    return error;
  }
  uint32_t src_offset = m0 & 1023;
  uint32_t dst_offset = (m0 >> 16) & 1023;
  uint64_t src = uint64_t(a_base) + src_offset;
  uint64_t dst = uint64_t(d_base) + dst_offset;
  if (!exec_mask || src_offset > 255 || dst_offset > 255 || src >= a_count || dst >= d_count) {
    return swap<Lanes>(flags, 0, mode, nullptr, nullptr);
  }
  return swap<Lanes>(flags, exec_mask, mode, d + dst, a + src);
}

// Swap VGPR values for the selected lanes. Reject invalid controls before changing outputs.
template <int Lanes>
int swap_half(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
              uint32_t *const *a) {
  const uint64_t known = RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (mode & ~known) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, 0, true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::swap_half_x86_64_v3(Lanes, exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned sd = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t old_d = (d[0][lane] >> sd) & 65535;
    uint32_t old_a = (a[0][lane] >> sa) & 65535;
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(65535U << sd)) | (old_a << sd);

      // Preserve the first write if A and D are halves of the same VGPR.
      a[0][lane] = (a[0][lane] & ~(65535U << sa)) | (old_d << sa);
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Exchange register values for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swap_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap<32>(flags, exec_mask, instruction_flags, d, a);
}

// Exchange register values for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swap_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap<64>(flags, exec_mask, instruction_flags, d, a);
}

// Exchange Wave64 half-waves.
// Return errors before changing outputs.
int rj_arith_v_permlane64_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *,
                                  const uint32_t *const *) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true);
}

// Exchange Wave64 half-waves for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlane64_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::permlane64_x86_64_v3(exec_mask, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  // Snapshot both halves before stores so D can alias A.
  uint32_t result[64];
  for (int lane = 0; lane < 64; ++lane) {
    result[lane] = a[0][lane ^ 32];
  }
  for (int lane = 0; lane < 64; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Exchange register values for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swap_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap_half<32>(flags, exec_mask, instruction_flags, d, a);
}

// Exchange register values for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swap_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap_half<64>(flags, exec_mask, instruction_flags, d, a);
}

// Exchange a register with its M0-relative peer for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swaprel_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap_relative<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base,
                           a_base, m0);
}

// Exchange a register with its M0-relative peer for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_swaprel_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swap_relative<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base,
                           a_base, m0);
}
