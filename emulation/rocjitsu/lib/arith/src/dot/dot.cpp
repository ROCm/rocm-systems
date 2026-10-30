// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 DOT2 entry points and their gfx11/gfx12 numerical models.
// Owns public entry points and their validation/dispatch policy.

#include "dot/dot.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "common/simd.h"
#include "dot/gfx11_dot2.h"
#include "dot/x86_64/dot2_exact.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Accumulate an FP16/BF16 DOT2 for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Bf16, bool Gfx11>
int dot(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, instruction_flags, a,
        [&](uint32_t exec_mask, const uint32_t *const *source) {
          return dot<Bf16, Gfx11>(flags, exec_mask, uint32_t(instruction_flags), d, source, b, c);
        });
  }

  // Bits 0..4: negation; bit 5: ABS_C; bit 6: CLAMP; bits 7..10: half selection.
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~0x7ffU)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (Bf16) {
    if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4) &&
        !rocjitsu::arith::is_exact(flags)) {
      rocjitsu::arith::dot2_bf16_x86_64_v4(exec_mask, uint32_t(instruction_flags), d[0], a[0], b[0],
                                           c[0]);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::is_exact(flags) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::dot2_exact_x86_64_v4(Bf16, Gfx11, exec_mask, instruction_flags, d[0], a[0],
                                          b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::is_exact(flags) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::dot2_exact_x86_64_v3(Bf16, Gfx11, exec_mask, instruction_flags, d[0], a[0],
                                          b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!rocjitsu::arith::is_exact(flags) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::dot2_x86_64_v3(Bf16, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  const int a0_shift = instruction_flags & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
  const int b0_shift = instruction_flags & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
  const int a1_shift = instruction_flags & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
  const int b1_shift = instruction_flags & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t left[2] = {uint16_t(a[0][lane] >> a0_shift), uint16_t(a[0][lane] >> a1_shift)};
    uint16_t right[2] = {uint16_t(b[0][lane] >> b0_shift), uint16_t(b[0][lane] >> b1_shift)};
    if (instruction_flags & RJ_ARITH_DOT_NEG_LO_A) {
      left[0] ^= 0x8000;
    }
    if (instruction_flags & RJ_ARITH_DOT_NEG_HI_A) {
      left[1] ^= 0x8000;
    }
    if (instruction_flags & RJ_ARITH_DOT_NEG_LO_B) {
      right[0] ^= 0x8000;
    }
    if (instruction_flags & RJ_ARITH_DOT_NEG_HI_B) {
      right[1] ^= 0x8000;
    }
    uint32_t acc_bits =
        (c[0][lane] & (instruction_flags & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U)) ^
        (instruction_flags & RJ_ARITH_DOT_NEG_C ? 0x80000000 : 0);
    if (rocjitsu::arith::is_exact(flags)) {
      if constexpr (!Gfx11) {
        result[lane] = rocjitsu::arith::gfx12_dot_bits<Bf16, 2>(left, right, acc_bits);
      } else {
        result[lane] =
            rocjitsu::arith::gfx11_dot2_f32<Bf16>(left[0], right[0], left[1], right[1], acc_bits);
      }
    } else {
      float acc = rocjitsu::arith::as_float(acc_bits);
      for (int j = 0; j < 2; ++j) {
        float x =
            Bf16 ? rocjitsu::arith::bf16_to_float(left[j]) : rocjitsu::arith::f16_to_float(left[j]);
        float y = Bf16 ? rocjitsu::arith::bf16_to_float(right[j])
                       : rocjitsu::arith::f16_to_float(right[j]);
        acc = __builtin_fmaf(x, y, acc);
      }
      result[lane] = rocjitsu::arith::as_bits(acc);
    }
  }

  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Accumulate an FP16/BF16 DOT2 for Wave64. Reject invalid controls before changing outputs.
template <bool Bf16, bool Gfx11>
int dot_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error =
          dot<Bf16, Gfx11>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags =
        rocjitsu::arith::dpp_half_wave_instruction_flags(instruction_flags, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = dot<Bf16, Gfx11>(flags, uint32_t(exec_mask >> (half * 32)),
                                     half_instruction_flags, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// RX 9070 DOT2 does not generate EXCP_FLAG_USER updates, including NaNs,
// denormals and overflow. Reporting leaves the register unchanged.
int rj_arith_v_dot2_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_bf16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // GFX11 result bits are characterized; its exception-register updates are not.
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // GFX11 result bits are characterized; its exception-register updates are not.
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2acc_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // The RDNA3 VOP2 form only encodes routing instruction_flags, unlike VOP3P DOT2.
  if (uint32_t(instruction_flags)) {
    return rocjitsu::arith::reports_fp_status(flags) ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                                     : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_f16_w32(flags, exec_mask, instruction_flags, d, a, b, d, trapsts);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_bf16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return dot_wave64<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // GFX11 result bits are characterized; its exception-register updates are not.
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2_f32_bf16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // GFX11 result bits are characterized; its exception-register updates are not.
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return dot_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products into D for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_dot2acc_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // The RDNA3 VOP2 form only encodes routing instruction_flags, unlike VOP3P DOT2.
  if (uint32_t(instruction_flags)) {
    return rocjitsu::arith::reports_fp_status(flags) ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                                     : RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rj_arith_v_dot2_f32_f16_w64(flags, exec_mask, instruction_flags, d, a, b, d, trapsts);
}
