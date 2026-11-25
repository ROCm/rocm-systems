// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA legacy/DX9 multiply with post-round flushing and guest output modifiers.
// Carries the old host-arithmetic policy without changing host FP controls.

#include "fp/legacy_mul.h"
#include "common/alu.h"
#include "common/validation.h"
#include "fp/fma_scalar.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

// Preserve legacy zero-product priority, quieting and post-round output flushing.
uint32_t rocjitsu::arith::legacy_mul_value(uint64_t flags, uint32_t mode, uint32_t a, uint32_t b) {
  a = alu_input_bits(a, mode, flags);
  b = alu_input_bits(b, mode >> 1, flags);
  const uint32_t ma = a & 0x7fffffffU, mb = b & 0x7fffffffU;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  uint32_t value = 0;
  if (ma && mb) {
    if (ma >= 0x7f800000U || mb >= 0x7f800000U) {
      value = as_bits(as_float(a) * as_float(b));
    } else {
      // Every finite FP32 product is exact in FP64, so narrowing explicitly
      // supplies guest rounding without relying on the host rounding mode.
      value =
          fma_narrow_f32(double_bits(double(as_float(a)) * double(as_float(b))), false, rounding);
    }
  }
  if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(value & 0x7f800000U)) {
    value &= 0x80000000U;
  }
  value = scale_omod_f32_bits(value, (mode >> 6) & 3, rounding);
  return mode & RJ_ARITH_ALU_CLAMP ? fma_clamp_f32(value, flags) : value;
}

namespace {

// Validate one wave block before access, then execute the inherited arithmetic.
template <bool Older>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  const uint32_t known = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A |
                         RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags |
                                                (Older ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t active, const uint32_t *const *source) {
          return run<Older>(flags, active, uint32_t(mode), d, source, b);
        });
  }
  if constexpr (Older) {
    mode = rocjitsu::arith::rdna3_output_modifiers(flags, uint32_t(mode));
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::legacy_mul_x86_64_v4(flags, exec_mask, uint32_t(mode), d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::legacy_mul_x86_64_v3(flags, exec_mask, uint32_t(mode), d[0], a[0], b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::legacy_mul_value(flags, uint32_t(mode), a[0][lane], b[0][lane]);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Validate once before either Wave64 block can write, keeping DPP row selection.
template <bool Older>
int wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = run<Older>(flags, 0, mode, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + 32 * half;
    const uint32_t *x = a[0] + 32 * half, *y = b[0] + 32 * half;
    const int error =
        run<Older>(flags, uint32_t(exec_mask >> (32 * half)),
                   rocjitsu::arith::dpp_half_wave_instruction_flags(mode, half), &dest, &x, &y);
    if (error) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply selected lanes with inherited gfx942/gfx950 policy; FP status remains unchanged.
int rj_arith_v_mul_legacy_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply selected lanes with inherited gfx942/gfx950 policy; FP status remains unchanged.
int rj_arith_v_mul_legacy_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  return wave64<true>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply selected lanes with inherited gfx1250 policy; FP status remains unchanged.
int rj_arith_v_mul_dx9_zero_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            [[maybe_unused]] uint32_t *excp_flag_user) {
  return run<false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Multiply selected lanes with inherited gfx1250 policy; FP status remains unchanged.
int rj_arith_v_mul_dx9_zero_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask,
                                            uint64_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            [[maybe_unused]] uint32_t *excp_flag_user) {
  return wave64<false>(flags, exec_mask, instruction_flags, d, a, b);
}
