// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DS lane swizzles using quad selectors or bitwise index transforms.
// Owns public entry points and their validation/dispatch policy.

// Basic swizzles follow rocjitsu. Rotate/FFT follow the common ISA pseudocode,
// checked against RX 9070 captures; RDNA3's masked-rotate examples disagree
// with its own pseudocode. Rotations wrap within each 32-lane row.

#include "common/internal.h"
#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Swizzle VGPR lanes for the selected lanes. Reject invalid controls before changing outputs.
template <int Lanes>
int swizzle(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
            const uint32_t *const *a, uint16_t offset) {
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
    rocjitsu::arith::ds_swizzle_x86_64_v3(Lanes, exec_mask, d[0], a[0], offset);
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned mask = offset & 31, mask_bits = 0;
  for (unsigned bit = 0; bit < 5; ++bit) {
    mask_bits += (mask >> bit) & 1;
  }
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    unsigned source;
    if (offset >= 0xe000) {
      unsigned reversed = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) |
                          ((lane & 16) >> 4);
      source = (reversed >> mask_bits) | (lane & mask);
    } else if (offset >= 0xc000) {
      unsigned rotate = (offset >> 5) & 31;
      if (offset & 1024) {
        rotate = 0U - rotate;
      }
      source = (lane & mask) | ((lane + rotate) & ~mask);
    } else if (offset & 0x8000) {
      source = (lane & ~3U) | ((offset >> (2 * (lane & 3))) & 3);
    } else {
      source = ((lane & mask) | ((offset >> 5) & 31)) ^ ((offset >> 10) & 31);
    }
    source = (lane & 32) | (source & 31);
    result[lane] = ((exec_mask >> source) & 1) ? a[0][source] : 0;
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Swizzle source VGPR lanes for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_swizzle_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swizzle<32>(flags, exec_mask, instruction_flags, d, a, offset);
}

// Swizzle source VGPR lanes for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_swizzle_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return swizzle<64>(flags, exec_mask, instruction_flags, d, a, offset);
}
