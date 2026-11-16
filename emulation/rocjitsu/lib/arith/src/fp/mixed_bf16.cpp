// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 mixed BF16 FMA: inherited rounding model, validation and dispatch.
// Finite BF16 results retain an exact FP64 product and the addition residual.

#include "fp/mixed_bf16.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <stdint.h>

namespace {

// Decode one mixed source without flushing denormals, then apply ABS/NEG.
uint32_t input(uint32_t raw, uint32_t mode) {
  if (mode & RJ_ARITH_MIX_BF16_A) {
    raw = (raw >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) << 16;
  }
  if (mode & RJ_ARITH_ALU_ABS_A) {
    raw &= 0x7fffffffU;
  }
  return raw ^ (mode & RJ_ARITH_ALU_NEG_A ? 0x80000000U : 0);
}

// Narrow finite FP64 bits to BF16 in the guest rounding mode. A round-to-odd
// intermediate preserves discarded information across the final narrowing.
uint16_t narrow(uint64_t bits, unsigned rounding) {
  const int exponent = int((bits >> 52) & 2047);
  const uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  const bool away = rounding == (sign ? 2U : 1U);
  if (exponent < 889) {
    return sign | (away && (bits & 0x7fffffffffffffffULL) ? 1 : 0);
  }
  if (exponent > 1150) {
    return sign | (rounding && !away ? 0x7f7f : 0x7f80);
  }
  const int shift = exponent < 897 ? 942 - exponent : 45;
  const uint64_t significand = (bits & 0xfffffffffffffULL) | (1ULL << 52);
  uint64_t result = significand >> shift;
  if (exponent >= 897) {
    result += uint64_t(exponent - 897) << 7;
  }
  const uint64_t remainder = significand & ((1ULL << shift) - 1);
  const uint64_t halfway = 1ULL << (shift - 1);
  result +=
      rounding ? away && remainder : remainder > halfway || (remainder == halfway && (result & 1));
  if (result >= 0x7f80) {
    result = rounding && !away ? 0x7f7f : 0x7f80;
  }
  return sign | uint16_t(result);
}

// Invoke one mixed FMA, rejecting invalid controls or strict modes before writes.
int run(rocjitsu::arith::MixedFma destination, uint64_t flags, uint32_t exec_mask, uint64_t mode,
        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
        const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                         RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_BF16_A | RJ_ARITH_MIX_BF16_B |
                         RJ_ARITH_MIX_BF16_C;
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known,
                                            rocjitsu::arith::kFmaGuestFpFlags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t active, const uint32_t *const *source) {
          return run(destination, flags, active, uint32_t(mode), d, source, b, c);
        });
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::mixed_bf16_x86_64_v4(destination, flags, exec_mask, uint32_t(mode), d[0], a[0],
                                          b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::mixed_bf16_x86_64_v3(destination, flags, exec_mask, uint32_t(mode), d[0], a[0],
                                          b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::mixed_bf16_value(
        destination, flags, mode & RJ_ARITH_ALU_CLAMP, input(a[0][lane], mode),
        input(b[0][lane], mode >> 1), input(c[0][lane], mode >> 2));
    if (destination != rocjitsu::arith::MixedFma::Float) {
      const int shift = destination == rocjitsu::arith::MixedFma::High ? 16 : 0;
      result[lane] = (d[0][lane] & ~(0xffffU << shift)) | (result[lane] << shift);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Port of fp_mode::detail::fma_f32_to_bf16_nearest_environment. The required
// default host environment supplies nearest arithmetic; no controls are written.
uint32_t rocjitsu::arith::mixed_bf16_value(MixedFma destination, uint64_t flags, bool clamp,
                                           uint32_t a, uint32_t b, uint32_t c) {
  const float x = as_float(a), y = as_float(b), z = as_float(c);
  const bool special = (a & 0x7fffffffU) >= 0x7f800000U || (b & 0x7fffffffU) >= 0x7f800000U ||
                       (c & 0x7fffffffU) >= 0x7f800000U;
  if (destination == MixedFma::Float || special) {
    float value = __builtin_fmaf(x, y, z);
    if (clamp) {
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    }
    if (destination == MixedFma::Float) {
      return as_bits(value);
    }
    // Preserve discarded NaN payload information in the narrowed result.
    const uint32_t bits = as_bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
      return (bits >> 16) | ((bits & 0xffffU) != 0);
    }
    return float_to_bf16(value);
  }
  // FP32 products fit exactly in FP64. Recover the sum's residual and jam an
  // even intermediate toward it, preserving BF16 midpoint and directed decisions.
  const double product = double(x) * double(y);
  const double sum = product + double(z);
  const double virtual_z = sum - product;
  const double error = (product - (sum - virtual_z)) + (double(z) - virtual_z);
  uint64_t bits = double_bits(sum);
  if (error != 0 && !(bits & 1)) {
    bits += ((bits ^ double_bits(error)) >> 63) ? -1ULL : 1ULL;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if (!(bits & 0x7fffffffffffffffULL)) {
    const bool negative_product = (a ^ b) >> 31;
    const bool matching_zeros = !(c & 0x7fffffffU) && negative_product == bool(c >> 31);
    bits = uint64_t(matching_zeros ? negative_product : rounding == 2) << 63;
  }
  const double value = as_double(bits);
  if (clamp) {
    if (value <= 0) {
      return 0;
    }
    if (value > 1) {
      return 0x3f80;
    }
  }
  return narrow(bits, rounding);
}

// Compute mixed FMA into FP32, preserving inactive lanes and returning errors before writes.
int rj_arith_v_fma_mix_f32_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  return run(rocjitsu::arith::MixedFma::Float, flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute mixed FMA into the low BF16 half, preserving inactive lanes and returning errors before
// writes.
int rj_arith_v_fma_mixlo_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  return run(rocjitsu::arith::MixedFma::Low, flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute mixed FMA into the high BF16 half, preserving inactive lanes and returning errors before
// writes.
int rj_arith_v_fma_mixhi_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  return run(rocjitsu::arith::MixedFma::High, flags, exec_mask, instruction_flags, d, a, b, c);
}
