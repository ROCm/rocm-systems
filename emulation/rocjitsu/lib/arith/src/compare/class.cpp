// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Floating-point class tests producing one predicate bit per lane.
// Owns public entry points and their validation/dispatch policy.

// Class bits follow rocjitsu's execute_v_cmp_class_f{16,32,64} bitwise models.

#include "compare/class.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>

namespace {

// Test floating-point classes for the selected lanes. Reject invalid controls before changing
// outputs.
template <int Bits>
int classify_wave32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 64) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
    if (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode)) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_CLAMP |
                         (Bits == 16 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B : 0);
  if (int error = rocjitsu::arith::validate(flags, uint32_t(mode) & ~known, true,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                RJ_ARITH_FP_IEEE_MODE)) {
    return error;
  }
  if (!exec_mask) {
    *d = 0;
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Bits != 64) {
    if (mode >> 32) {
      uint32_t permuted[32];
      const uint32_t *source = permuted;
      if (mode & RJ_ARITH_DPP8) {
        rocjitsu::arith::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
      } else {
        exec_mask = rocjitsu::arith::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
      }
      return classify_wave32<Bits>(flags, exec_mask, uint32_t(mode), d, &source, b);
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    *d = rocjitsu::arith::class_x86_64_v4<Bits>(mode, a, b[0]) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    *d = rocjitsu::arith::class_x86_64_v3<Bits>(mode, a, b[0]) & exec_mask;
    return RJ_ARITH_SUCCESS;
  }
#endif
  // Classification uses encoded bits, so sNaNs are identified without FP evaluation.
  constexpr unsigned fraction_bits = Bits == 16 ? 10 : Bits == 32 ? 23 : 20;
  constexpr uint32_t exponent_mask = Bits == 16 ? 31 : Bits == 32 ? 255 : 2047;
  constexpr uint32_t sign = Bits == 16 ? 0x8000 : 0x80000000;
  uint32_t result = 0;

  // Clang lowers vectorized variable shifts to FP conversions on baseline x86,
  // raising exception flags even with FENV_ACCESS enabled. Keep this path scalar.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t high = a[Bits == 64 ? 1 : 0][lane], classes = b[0][lane];
    if constexpr (Bits == 16) {
      high = (high >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
      classes >>= mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
    }
    if (mode & RJ_ARITH_ALU_ABS_A) {
      high &= ~sign;
    }
    if (mode & RJ_ARITH_ALU_NEG_A) {
      high ^= sign;
    }
    uint32_t exponent = (high >> fraction_bits) & exponent_mask;
    uint32_t fraction = high & ((1u << fraction_bits) - 1);
    if constexpr (Bits == 64) {
      fraction |= a[0][lane];
    }
    int index;
    if (exponent == exponent_mask && fraction) {
      index = (high >> (fraction_bits - 1)) & 1;
    } else {
      index = exponent == exponent_mask ? 9 : exponent ? 8 : fraction ? 7 : 6;
      if (high & sign) {
        index = 11 - index;
      }
    }
    result |= ((classes >> index) & 1) << lane;
  }
  *d = result & exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Test floating-point classes for Wave64. Reject invalid controls before changing outputs.
template <int Bits>
int classify_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint64_t *d,
                    const uint32_t *const *a, const uint32_t *const *b) {
  uint32_t masks[2] = {};

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = classify_wave32<Bits>(flags, 0, mode, masks, nullptr, nullptr)) {
    return error;
  }
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint64_t half_mode = mode;
      if (mode & RJ_ARITH_DPP16) {
        uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
        half_mode = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                    (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
      }
      constexpr int words = Bits == 64 ? 2 : 1;
      const uint32_t *source_a[words], *source_b = b[0] + half * 32;
      for (int word = 0; word < words; ++word) {
        source_a[word] = a[word] + half * 32;
      }
      if (int error = classify_wave32<Bits>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                            masks + half, source_a, &source_b)) {
        return error;
      }
    }
  }
  uint64_t result = masks[0] | (uint64_t(masks[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  std::memcpy(static_cast<void *>(d), &result, sizeof result);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<16>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<32>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<64>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<16>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<32>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave32 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave32<64>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<16>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<32>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmp_class_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<64>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<16>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<32>(flags, exec_mask, instruction_flags, d, a, b);
}

// Write the Wave64 mask of A values whose FP class is selected by B.
// Errors preserve outputs.
int rj_arith_v_cmpx_class_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint64_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return classify_wave64<64>(flags, exec_mask, instruction_flags, d, a, b);
}
