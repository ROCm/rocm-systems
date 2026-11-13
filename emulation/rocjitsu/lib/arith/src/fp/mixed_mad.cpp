// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA mixed MAD retains an FP32 arithmetic intermediate before half narrowing.
// Approximate inherited semantics; guest denormal controls do not flush operands.
#include "fp/mixed_mad.h"
#include "common/validation.h"
#include "fp/fma_scalar.h"
#include "fp/mixed_fma_scalar.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

// Preserve the inherited FP32 intermediate and apply CLAMP before half narrowing.
uint32_t rocjitsu::arith::mixed_mad_value(MixedFma destination, uint64_t flags, uint32_t mode,
                                          uint32_t a, uint32_t b, uint32_t c) {
  float x = as_float(mixed_fma_input(a, mode));
  float y = as_float(mixed_fma_input(b, mode >> 1));
  float z = as_float(mixed_fma_input(c, mode >> 2));
  // Explicit FP32 fusion retains the old optimized scalar arithmetic without
  // making its result depend on compiler expression-contraction choices.
  uint32_t bits = as_bits(__builtin_fmaf(x, y, z));
  if (mode & RJ_ARITH_ALU_CLAMP) {
    bits = fma_clamp_f32(bits, flags);
  }
  if (destination == MixedFma::Float) {
    return bits;
  }
  if ((bits & 0x7fffffffU) >= 0x7f800000U) {
    return float_to_f16(as_float(bits), false);
  }
  return half_fma_narrow(double_bits(double(as_float(bits))), flags & RJ_ARITH_FP16_OVFL,
                         unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
}

namespace {

// Validate before any output write, then dispatch one or two Wave32 blocks.
template <rocjitsu::arith::MixedFma Destination, typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                         RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B |
                         RJ_ARITH_MIX_F16_C;
  if (int error = validate(flags, mode & ~uint64_t(known),
                           kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF |
                               RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK)) {
    return error;
  }
  if ((flags & RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) == RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    const uint32_t active = uint32_t(exec_mask >> offset);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      mixed_mad_x86_64_v4(Destination, flags, active, uint32_t(mode), d[0] + offset, a[0] + offset,
                          b[0] + offset, c[0] + offset);
      continue;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      mixed_mad_x86_64_v3(Destination, flags, active, uint32_t(mode), d[0] + offset, a[0] + offset,
                          b[0] + offset, c[0] + offset);
      continue;
    }
#endif
    uint32_t result[32];
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = mixed_mad_value(Destination, flags, uint32_t(mode), a[0][offset + lane],
                                     b[0][offset + lane], c[0][offset + lane]);
    }
    for (int lane = 0; lane < 32; ++lane) {
      if ((active >> lane) & 1) {
        if constexpr (Destination == MixedFma::Float) {
          d[0][offset + lane] = result[lane];
        } else {
          constexpr int shift = Destination == MixedFma::High ? 16 : 0;
          d[0][offset + lane] =
              (d[0][offset + lane] & ~(0xffffU << shift)) | (result[lane] << shift);
        }
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mix_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mix_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mixlo_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mixlo_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mixhi_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected mixed MAD lanes; errors leave destination and FP status unchanged.
int rj_arith_v_mad_mixhi_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d, a, b, c);
}
