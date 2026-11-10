// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bitfield extraction/insertion and alignment of adjacent source words.
// Owns public entry points and their validation/dispatch policy.

#include "integer/bitfield.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Extract, insert, or align bitfields.
template <rocjitsu::arith::Bitfield Op> uint32_t evaluate(uint32_t a, uint32_t b, uint32_t c) {
  if constexpr (Op == rocjitsu::arith::Bitfield::ExtractUnsigned ||
                Op == rocjitsu::arith::Bitfield::ExtractSigned) {
    unsigned offset = b & 31, width = c & 31;
    uint32_t field_mask = (uint32_t(1) << width) - 1;
    if constexpr (Op == rocjitsu::arith::Bitfield::ExtractUnsigned) {
      return (a >> offset) & field_mask;
    }

    // rocjitsu's simd_bfe_i32 sign-extends A before extracting: fields
    // crossing bit 31 include copies of A's sign bit.
    uint32_t sign = 0u - (a >> 31);
    uint32_t field = (((a ^ sign) >> offset) ^ sign) & field_mask;
    uint32_t sign_bit = (uint32_t(1) << width) >> 1;
    return (field ^ sign_bit) - sign_bit;
  }
  if constexpr (Op == rocjitsu::arith::Bitfield::AlignBit ||
                Op == rocjitsu::arith::Bitfield::AlignByte) {
    unsigned shift = Op == rocjitsu::arith::Bitfield::AlignBit ? c & 31 : (c & 3) * 8;
    return uint32_t(((uint64_t(a) << 32) | b) >> shift);
  }
  if constexpr (Op == rocjitsu::arith::Bitfield::Permute) {
    // Byte selection and sign-fill semantics follow rocjitsu's perm_b32.
    uint64_t source = (uint64_t(a) << 32) | b;
    uint32_t result = 0;
    for (int byte = 0; byte < 4; ++byte) {
      unsigned selector = (c >> (8 * byte)) & 255;
      uint32_t value;
      if (selector < 8) {
        value = uint32_t(source >> (8 * selector)) & 255;
      } else if (selector < 12) {
        value = ((source >> (16 * (selector - 8) + 15)) & 1) ? 255 : 0;
      } else {
        value = selector == 12 ? 0 : 255;
      }
      result |= value << (8 * byte);
    }
    return result;
  }
  if constexpr (Op == rocjitsu::arith::Bitfield::Insert) {
    return (a & b) | (~a & c);
  }
  if constexpr (Op == rocjitsu::arith::Bitfield::Mask) {
    return ((uint32_t(1) << (a & 31)) - 1) << (b & 31);
  }
  if constexpr (Op == rocjitsu::arith::Bitfield::Reverse) {
    return rocjitsu::arith::bit_reverse(a);
  }
}

// Extract, insert, or align bitfields for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Bitfield Op>
int bitfield(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return bitfield<Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }

  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(RJ_ARITH_ALU_CLAMP))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *bp = nullptr, *cp = nullptr;
  if constexpr (Op != rocjitsu::arith::Bitfield::Reverse) {
    bp = b[0];
  }
  if constexpr (Op != rocjitsu::arith::Bitfield::Reverse && Op != rocjitsu::arith::Bitfield::Mask) {
    cp = c[0];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::bitfield_x86_64_v4<Op>(exec_mask, d[0], a[0], bp, cp);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::bitfield_x86_64_v3<Op>(exec_mask, d[0], a[0], bp, cp);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  if constexpr (Op == rocjitsu::arith::Bitfield::ExtractUnsigned ||
                Op == rocjitsu::arith::Bitfield::ExtractSigned) {
    // Clang's baseline vector shifts use FP conversions for powers of two;
    // converting 2^31 raises FE_INVALID. Prevent both loop and SLP vectorization.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = evaluate<Op>(a[0][lane], bp[lane], cp[lane]);
    }
  } else {
    for (int lane = 0; lane < 32; ++lane) {
      result[lane] = evaluate<Op>(a[0][lane], bp ? bp[lane] : 0, cp ? cp[lane] : 0);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Extract, insert, or align bitfields for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Bitfield Op>
int bitfield_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = bitfield<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = nullptr, *source_c = nullptr;
    if constexpr (Op != rocjitsu::arith::Bitfield::Reverse) {
      source_b = b[0] + half * 32;
    }
    if constexpr (Op != rocjitsu::arith::Bitfield::Reverse &&
                  Op != rocjitsu::arith::Bitfield::Mask) {
      source_c = c[0] + half * 32;
    }
    if (int error = bitfield<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_instruction_flags,
                                 &dest, &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Extract a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfe_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::ExtractUnsigned>(flags, exec_mask, instruction_flags,
                                                              d, a, b, c);
}

// Extract a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfe_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::ExtractSigned>(flags, exec_mask, instruction_flags, d,
                                                            a, b, c);
}

// Insert a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfi_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::Insert>(flags, exec_mask, instruction_flags, d, a, b,
                                                     c);
}

// Construct a bitfield mask for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfm_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::Mask>(flags, exec_mask, instruction_flags, d, a, b,
                                                   nullptr);
}

// Reverse the bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::Reverse>(flags, exec_mask, instruction_flags, d, a,
                                                      nullptr, nullptr);
}

// Align concatenated source words by a bit offset for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_alignbit_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::AlignBit>(flags, exec_mask, instruction_flags, d, a, b,
                                                       c);
}

// Align concatenated source words by a byte offset for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_alignbyte_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::AlignByte>(flags, exec_mask, instruction_flags, d, a,
                                                        b, c);
}

// Permute source bytes for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_perm_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield<rocjitsu::arith::Bitfield::Permute>(flags, exec_mask, instruction_flags, d, a, b,
                                                      c);
}

// Extract a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfe_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::ExtractUnsigned>(flags, exec_mask,
                                                                     instruction_flags, d, a, b, c);
}

// Extract a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfe_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::ExtractSigned>(flags, exec_mask,
                                                                   instruction_flags, d, a, b, c);
}

// Insert a bitfield for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfi_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::Insert>(flags, exec_mask, instruction_flags, d,
                                                            a, b, c);
}

// Construct a bitfield mask for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfm_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::Mask>(flags, exec_mask, instruction_flags, d, a,
                                                          b, nullptr);
}

// Reverse the bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bfrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::Reverse>(flags, exec_mask, instruction_flags, d,
                                                             a, nullptr, nullptr);
}

// Align concatenated source words by a bit offset for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_alignbit_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::AlignBit>(flags, exec_mask, instruction_flags,
                                                              d, a, b, c);
}

// Align concatenated source words by a byte offset for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_alignbyte_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::AlignByte>(flags, exec_mask, instruction_flags,
                                                               d, a, b, c);
}

// Permute source bytes for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_perm_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bitfield_wave64<rocjitsu::arith::Bitfield::Permute>(flags, exec_mask, instruction_flags, d,
                                                             a, b, c);
}
