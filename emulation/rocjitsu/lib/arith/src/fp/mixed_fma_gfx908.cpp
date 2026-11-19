// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx908 mixed MAD composes guest-controlled FP32 FMA and optional half narrowing.
// Reuses the existing arithmetic model and SIMD backends without host FP changes.
#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/half_fma_scalar.h"
#include "fp/mixed_mad.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

namespace {

// Validate both stages before writing any lane; preserve unselected destination data.
template <rocjitsu::arith::MixedFma Destination, typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE);
  constexpr uint64_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                             RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                             RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                             RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B |
                             RJ_ARITH_MIX_F16_C;
  constexpr bool half = Destination != MixedFma::Float;
  if (int error = validate(flags, mode & ~known,
                           kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF |
                               (half ? RJ_ARITH_FP_ROUND_INTERMEDIATE_F32_MASK : 0))) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // Source halves survive MODE input flushing. Only true FP32 inputs use that field.
  uint64_t arithmetic_flags =
      (flags & ~(RJ_ARITH_FP_ROUND_INTERMEDIATE_F32_MASK | RJ_ARITH_FP16_OVFL)) |
      RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16;
  if constexpr (half) {
    arithmetic_flags = (arithmetic_flags & ~RJ_ARITH_FP_ROUND_MASK) |
                       ((flags & RJ_ARITH_FP_ROUND_INTERMEDIATE_F32_MASK) >> 4);
  }
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    const uint32_t active = uint32_t(exec_mask >> offset);
    if (!active) {
      continue;
    }
    const uint32_t *x = a[0] + offset, *y = b[0] + offset, *z = c[0] + offset;
    uint32_t intermediate[32]{};
    uint32_t *destination = half ? intermediate : d[0] + offset;
    // The established FP32 path owns guest rounding, denormals, NaNs and CLAMP.
    if (int error = rj_arith_v_fma_mix_f32_w32(arithmetic_flags, active, mode, &destination, &x, &y,
                                               &z, nullptr)) {
      return error;
    }
    if constexpr (half) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
      if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
        mixed_narrow_x86_64_v4(Destination, flags, active, d[0] + offset, intermediate);
        continue;
      }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
        mixed_narrow_x86_64_v3(Destination, flags, active, d[0] + offset, intermediate);
        continue;
      }
#endif
      constexpr int shift = Destination == MixedFma::High ? 16 : 0;
      const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
      for (int lane = 0; lane < 32; ++lane) {
        if ((active >> lane) & 1) {
          const uint32_t bits = intermediate[lane];
          // Quiet before truncating the payload, matching native SIMD conversion.
          const uint16_t result =
              (bits & 0x7fffffffU) >= 0x7f800000U
                  ? float_to_f16(as_float(bits | ((bits & 0x7fffffU) ? 0x00400000U : 0)))
                  : half_fma_narrow(double_bits(double(as_float(bits))), flags & RJ_ARITH_FP16_OVFL,
                                    rounding);
          uint32_t &word = d[0][offset + lane];
          word = (word & ~(0xffffU << shift)) | (uint32_t(result) << shift);
        }
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mix_f32_w32_gfx908(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mix_f32_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c,
                                      [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Float>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mixlo_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mixlo_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::Low>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mixhi_f16_w32_gfx908(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate selected lanes using the inherited two-stage gfx908 numeric model.
int rj_arith_v_mad_mixhi_f16_w64_gfx908(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return run<rocjitsu::arith::MixedFma::High>(flags, exec_mask, instruction_flags, d, a, b, c);
}
