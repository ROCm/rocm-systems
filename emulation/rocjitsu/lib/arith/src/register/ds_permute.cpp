// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DS gather/scatter lane permutations over VGPR data.
// Owns public entry points and their validation/dispatch policy.

// Register routing follows rocjitsu's DS handlers. RDNA3 Wave64 routes within
// each half-wave; RDNA4 Wave64 routes across the entire wave.

#include "common/internal.h"
#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Route VGPR words by per-lane addresses for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Backward, int Lanes, int Group, bool ReadInactive = false>
int permute(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
            const uint32_t *const *addr, const uint32_t *const *data, uint16_t offset) {
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (Backward && !ReadInactive) {
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      rocjitsu::arith::ds_bpermute_x86_64_v3(Lanes, Group, exec_mask, d[0], addr[0], data[0],
                                             offset);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  uint32_t result[Lanes] = {};
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t address = addr[0][lane] + offset;
    int selected = (lane & ~(Group - 1)) | ((address >> 2) & (Group - 1));
    if constexpr (Backward) {
      result[lane] = (ReadInactive || ((exec_mask >> selected) & 1)) ? data[0][selected] : 0;
    } else {
      // Visit every lane in order. The highest active source wins collisions,
      // matching rocjitsu and the ISA pseudocode's deterministic choice.
      if ((exec_mask >> lane) & 1) {
        result[selected] = data[0][lane];
      }
    }
  }
  // Delay stores until every source/address has been consumed, including aliases.
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Scatter VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_permute_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *addr,
                                const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<false, 32, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Scatter VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_permute_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *addr,
                                const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<false, 64, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Scatter VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_permute_b32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *addr, const uint32_t *const *data,
                                      uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<false, 64, 64>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Gather VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_bpermute_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *addr,
                                 const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<true, 32, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Gather VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_bpermute_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *addr,
                                 const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<true, 64, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Gather VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_bpermute_b32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *addr, const uint32_t *const *data,
                                       uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<true, 64, 64>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Gather VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_bpermute_fi_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<true, 32, 32, true>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

// Gather VGPR lanes by address for selected lanes.
// Return errors before changing outputs.
int rj_arith_ds_bpermute_fi_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return permute<true, 64, 64, true>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}
