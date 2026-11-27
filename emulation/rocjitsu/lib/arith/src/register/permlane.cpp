// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lane permutations across rows, half-waves, and whole waves.
// Owns public entry points and their validation/dispatch policy.

// Lane selection and inactive-source rules follow rocjitsu's RDNA4 VOP3 models.

#include "register/permlane.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace {

// Permute source lanes for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Cross, bool Var>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t lo, uint32_t hi) {
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~(RJ_ARITH_PERMLANE_FI | RJ_ARITH_PERMLANE_BOUND_CTRL),
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *indices = Var ? b[0] : nullptr;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::permlane_x86_64_v4<Cross, Var>(exec_mask, mode, d[0], a[0], indices, lo, hi);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
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

// Preserve inherited group normalization and uint32 selector arithmetic. Sources
// ignore EXEC, out-of-group indices retain the same lane, and writes may alias the source.
template <int Lanes>
int group_permutation(rocjitsu::arith::GroupPermutation op, uint64_t flags, uint64_t exec_mask,
                      uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
                      uint32_t selector, uint32_t width) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (!width || (width & (width - 1)) || width > Lanes) {
    width = Lanes;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::group_permutation_x86_64_v4(op, Lanes, exec_mask, d[0], a[0], selector, width);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::group_permutation_x86_64_v3(op, Lanes, exec_mask, d[0], a[0], selector, width);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t offset = uint32_t(lane) & (width - 1);
    uint32_t base = uint32_t(lane) - offset, source = lane;
    using Op = rocjitsu::arith::GroupPermutation;
    switch (op) {
    case Op::Bcast:
      source = base + (selector & (width - 1));
      break;
    case Op::Up:
      source = selector <= offset ? lane - selector : lane;
      break;
    case Op::Down:
      source = selector < width - offset ? lane + selector : lane;
      break;
    case Op::Xor:
      // XOR may read an earlier group; only the upper group boundary is a limit.
      source = selector < Lanes && (uint32_t(lane) ^ selector) < base + width
                   ? uint32_t(lane) ^ selector
                   : uint32_t(lane);
      break;
    }
    result[lane] = a[0][source];
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Generate byte offsets within the selected group from the low six input bits.
template <int Lanes>
int index_generation(uint64_t flags, uint64_t exec_mask, uint64_t controls, uint32_t *const *d,
                     const uint32_t *const *a, uint32_t group_width) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = validate(flags, controls)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    permlane_index_x86_64_v4(Lanes, exec_mask, d[0], a[0], group_width);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    permlane_index_x86_64_v3(Lanes, exec_mask, d[0], a[0], group_width);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    result[lane] = ((uint32_t(lane) & ~(group_width - 1)) | (a[0][lane] & 63 & (group_width - 1)))
                   << 2;
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
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

// Validate flags for the Wave32 no-op.
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
  return rocjitsu::arith::validate(flags, uint32_t(instruction_flags));
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
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
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

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_bcast_b32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, uint32_t selector,
                                      uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<32>(rocjitsu::arith::GroupPermutation::Bcast, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_bcast_b32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, uint32_t selector,
                                      uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<64>(rocjitsu::arith::GroupPermutation::Bcast, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_up_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t selector,
                                   uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<32>(rocjitsu::arith::GroupPermutation::Up, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_up_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t selector,
                                   uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<64>(rocjitsu::arith::GroupPermutation::Up, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_down_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t selector, uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<32>(rocjitsu::arith::GroupPermutation::Down, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_down_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     uint32_t selector, uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<64>(rocjitsu::arith::GroupPermutation::Down, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_xor_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a, uint32_t selector,
                                    uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<32>(rocjitsu::arith::GroupPermutation::Xor, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Route within lane groups, preserving inactive destinations and source aliases.
int rj_arith_v_permlane_xor_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a, uint32_t selector,
                                    uint32_t group_width) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return group_permutation<64>(rocjitsu::arith::GroupPermutation::Xor, flags, exec_mask,
                               instruction_flags, d, a, selector, group_width);
}

// Generate indices for active lanes, allowing destination/source overlap.
int rj_arith_v_permlane_idx_gen_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t controls,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        uint32_t group_width) {
  return index_generation<32>(flags, exec_mask, controls, d, a, group_width);
}

// Generate indices for active lanes, allowing destination/source overlap.
int rj_arith_v_permlane_idx_gen_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t controls,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        uint32_t group_width) {
  return index_generation<64>(flags, exec_mask, controls, d, a, group_width);
}
