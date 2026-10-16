// SPDX-License-Identifier: MIT

// Division operand rescaling and its per-lane scaling-condition output.
// Owns public entry points and their validation/dispatch policy.

#include "fp/div_scale.h"
#include "common/internal.h"
#include "fp/division.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>

namespace {

// Rescale division operands and produce scaling conditions for the selected lanes. Reject
// invalid controls before changing outputs.
template <int Width, bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d, uint32_t *condition,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;

  // RDNA3 ignores OMOD with preserved output denormals or IEEE enabled.
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      mode &= ~RJ_ARITH_ALU_OMOD_HALF;
    }
  }

  // The borrowed RDNA3 model covers unscaled results, not active OMOD.
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known, !Rdna3 || !(mode & RJ_ARITH_ALU_OMOD_HALF),
          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
              RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_ROUND_MASK)) {
    return error;
  }
  if (!exec_mask) {
    *condition = 0;
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    *condition = rocjitsu::arith::div_scale_x86_64_v4<Width>(exec_mask, mode, d, a, b, c,
                                                             Width == 32 || !Rdna3, flags);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    *condition = rocjitsu::arith::div_scale_x86_64_v3<Width>(exec_mask, mode, d, a, b, c,
                                                             Width == 32 || !Rdna3, flags);
    return RJ_ARITH_SUCCESS;
  }
#endif
  using T = typename rocjitsu::arith::DivisionFormat<Width>::Bits;
  T result[32];
  uint32_t output_mask = 0;
  for (int lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    auto scaled =
        rocjitsu::arith::division_scale<Width>(av, bv, cv, mode, Width == 32 || !Rdna3, flags);
    result[lane] = scaled.value;
    output_mask |= uint32_t(scaled.post_scale) << lane;
  }
  for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
      }
    }
  }
  *condition = output_mask & exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Rescale division operands and produce scaling conditions for Wave64. Reject invalid controls
// before changing outputs.
template <int Width, bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               uint64_t *condition, const uint32_t *const *a, const uint32_t *const *b,
               const uint32_t *const *c) {
  uint32_t half_condition = 0;
  if (int error =
          run<Width, Rdna3>(flags, 0, mode, nullptr, &half_condition, nullptr, nullptr, nullptr)) {
    return error;
  }
  uint64_t result_condition = 0;
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint32_t *dest[Width / 32];
      const uint32_t *av[Width / 32], *bv[Width / 32], *cv[Width / 32];
      for (int reg = 0; reg < Width / 32; ++reg) {
        dest[reg] = d[reg] + half * 32;
        av[reg] = a[reg] + half * 32;
        bv[reg] = b[reg] + half * 32;
        cv[reg] = c[reg] + half * 32;
      }
      if (int error = run<Width, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest,
                                        &half_condition, av, bv, cv)) {
        return error;
      }
      result_condition |= uint64_t(half_condition) << (half * 32);
    }
  }
  // Commit after all operand reads and VGPR writes, including aliased storage.
  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  std::memcpy(static_cast<void *>(condition), &result_condition, sizeof(result_condition));
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Execute DIV_SCALE_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint32_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<32>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F64 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint32_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<64>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint64_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<32>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F64 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       uint64_t *condition, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<64>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F64 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_HALF) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<64, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F64 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_HALF) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_HALF) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<32, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

// Execute DIV_SCALE_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_div_scale_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, (!(instruction_flags & RJ_ARITH_ALU_OMOD_HALF) || (flags & RJ_ARITH_FP_IEEE_MODE) ||
              !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))
                 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                 : RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<32, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}
