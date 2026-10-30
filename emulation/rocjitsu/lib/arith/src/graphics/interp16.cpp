// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 interpolation with packed source selection and staged rounding.
// Owns public entry points and their validation/dispatch policy.

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fp64_impl.h"
#include "fp/mixed_fma_scalar.h"
#include "graphics/interp.h"
#include "graphics/interp16_scalar.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Preserve discarded bits without using host floating-point arithmetic.
static double interp16_round_odd(uint32_t a, uint32_t b, uint32_t c) {
  return as_double(fma_finite_round_odd(a, b, c));
}

// Evaluate the interpolation stage with round-to-zero FP32 intermediate arithmetic.
float interp16_rtz_float(uint32_t a, uint32_t b, uint32_t c) {
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    return __builtin_fmaf(as_float(a), as_float(b), as_float(c));
  }
  uint64_t bits = fma_finite_round_odd(a, b, c);
  uint32_t sign = uint32_t(bits >> 32) & 0x80000000;
  int exponent = int((bits >> 52) & 2047) - 1023;
  uint64_t significand = (bits & 0xfffffffffffffULL) | (1ULL << 52);
  if (exponent > 127) {
    return as_float(sign | 0x7f7fffff);
  }
  if (exponent < -149) {
    return as_float(sign);
  }
  if (exponent < -126) {
    return as_float(sign | uint32_t(significand >> (-exponent - 97)));
  }
  return as_float(sign | (uint32_t(exponent + 127) << 23) |
                  (uint32_t(significand >> 29) & 0x7fffff));
}

// Evaluate the interpolation stage with round-to-zero FP16 output.
static uint16_t interp16_rtz_half(uint32_t a, uint32_t b, uint32_t c, bool clamp) {
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    result = mixed_fma_special(a, b, c);
  } else {
    uint64_t bits = double_bits(interp16_round_odd(a, b, c));
    unsigned exponent = unsigned((bits >> 52) & 2047);
    uint16_t sign = uint16_t((bits >> 48) & 0x8000);
    uint64_t significand = (bits & 0xfffffffffffffULL) | (1ULL << 52);
    if (exponent < 999) {
      result = sign;
    } else if (exponent > 1038) {
      result = sign | 0x7bff;
    } else if (exponent < 1009) {
      result = sign | uint16_t(significand >> (1051 - exponent));
    } else {
      result = sign | uint16_t(((exponent - 1009) << 10) + (significand >> 42));
    }
  }
  return clamp ? half_fma_clamp(result) : result;
}

} // namespace rocjitsu::arith

namespace {

// Evaluate an FP16 interpolation stage for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool P2, bool Rtz>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A |
                         (P2 ? RJ_ARITH_ALU_HIGH_D : RJ_ARITH_ALU_HIGH_C) |
                         RJ_ARITH_INTERP_WAIT_EXP_MASK;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::interp16_x86_64_v3<P2, Rtz>(flags & RJ_ARITH_FP16_OVFL, exec_mask, mode, d[0],
                                                 a[0], b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sc = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0,
           sd = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    // Interpolation coefficients come from fixed lanes of each quad; B and the P2 accumulator
    // remain per-lane.
    unsigned quad = lane & ~3u;
    uint32_t x = rocjitsu::arith::as_bits(
                 rocjitsu::arith::f16_to_float(uint16_t(a[0][quad + (P2 ? 2 : 1)] >> sa))),
             y = b[0][lane];
    uint32_t z =
        P2 ? c[0][lane]
           : rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(uint16_t(c[0][quad] >> sc)));
    if (mode & RJ_ARITH_ALU_NEG_A) {
      x ^= 0x80000000;
    }
    if (mode & RJ_ARITH_ALU_NEG_B) {
      y ^= 0x80000000;
    }
    if (mode & RJ_ARITH_ALU_NEG_C) {
      z ^= 0x80000000;
    }
    if constexpr (P2) {
      uint16_t h = Rtz ? rocjitsu::arith::interp16_rtz_half(x, y, z, mode & RJ_ARITH_ALU_CLAMP)
                       : rocjitsu::arith::mixed_fma_half_value(x, y, z, mode & RJ_ARITH_ALU_CLAMP,
                                                               flags & RJ_ARITH_FP16_OVFL);
      result[lane] = (d[0][lane] & ~(65535u << sd)) | (uint32_t(h) << sd);
    } else {
      float value = Rtz ? rocjitsu::arith::interp16_rtz_float(x, y, z)
                        : __builtin_fmaf(rocjitsu::arith::as_float(x), rocjitsu::arith::as_float(y),
                                         rocjitsu::arith::as_float(z));
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = !(value > 0) ? 0 : value > 1 ? 1 : value;
      }
      result[lane] = rocjitsu::arith::as_bits(value);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate an FP16 interpolation stage for Wave64. Reject invalid controls before changing
// outputs.
template <bool P2, bool Rtz>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<P2, Rtz>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
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
            run<P2, Rtz>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest, &av, &bv, &cv)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Execute INTERP_P10_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P10_RTZ_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_RTZ_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_rtz_f16_f32_w32(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P10_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P10_RTZ_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p10_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Execute INTERP_P2_RTZ_F16_F32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_interp_p2_rtz_f16_f32_w64(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
