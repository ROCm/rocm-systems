// SPDX-License-Identifier: MIT

// Lane permutations across rows, half-waves, and whole waves.
// Owns public entry points and their validation/dispatch policy.

// Lane selection and inactive-source rules follow rocjitsu's RDNA4 VOP3 models.

#include "register/permlane.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Permute source lanes for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Cross, bool Var>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t lo, uint32_t hi) {
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL), true,
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *indices = Var ? b[0] : nullptr;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::permlane_x86_64_v4<Cross, Var>(exec_mask, mode, d[0], a[0], indices, lo, hi);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::permlane_x86_64_v3<Cross, Var>(exec_mask, mode, d[0], a[0], indices, lo, hi);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    unsigned sel = Var ? indices[lane] : ((lane & 8 ? hi : lo) >> ((lane & 7) * 4));
    int source = ((lane & 16) ^ (Cross ? 16 : 0)) | (sel & 15);
    bool readable = ((exec_mask >> source) & 1) || (mode & RJ_ARITH_PERMLANE_FI);
    result[lane] = readable ? a[0][source] : (mode & RJ_ARITH_PERMLANE_BOUND_CTRL ? 0 : d[0][lane]);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Row pairs never cross a 32-lane boundary, including in the X forms.
template <bool Cross, bool Var>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, uint32_t lo, uint32_t hi) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Cross, Var>(flags, 0, mode, nullptr, nullptr, nullptr, lo, hi)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    const uint32_t *indices = Var ? b[0] + half * 32 : nullptr;
    if (int error = run<Cross, Var>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest, &source,
                                    &indices, lo, hi)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Permute lanes within rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlane16_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                  uint32_t hi) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

// Permute lanes across rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlanex16_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                   uint32_t hi) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

// Permute lanes within rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlane16_var_b32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}

// Permute lanes across rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlanex16_var_b32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}

// Permute lanes within rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlane16_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                  uint32_t hi) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<false, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

// Permute lanes within rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlane16_var_b32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}

// Permute lanes across rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlanex16_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                   uint32_t hi) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<true, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

// Permute lanes across rows for selected lanes.
// Return errors before changing outputs.
int rj_arith_v_permlanex16_var_b32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}
