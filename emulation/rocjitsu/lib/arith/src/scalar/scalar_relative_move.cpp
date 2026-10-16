// SPDX-License-Identifier: MIT

// M0-relative scalar register reads/writes with bounds validation.
// Owns public entry points and their validation/dispatch policy.

// Both ISA manuals, sections 3.3.1 and S_MOVREL: indices count DWORDs,
// and indexing cannot cross between SGPR/VCC and trap-temporary regions.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Validate the instruction's accepted flags before operand access or output writes.
int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return rocjitsu::arith::validate(flags, uint32_t(mode), true);
}

// Return whether the requested register range lies within the supplied register file.
bool in_range(uint32_t base, uint32_t offset, uint32_t count, int words) {
  uint64_t end = uint64_t(base) + offset + words;
  unsigned region_end = base < 108 ? 108 : base < 124 ? 124 : 0;
  return end <= region_end && end <= count;
}

// Read the selected scalar register value, including the instruction's special-register
// mapping.
uint64_t read(const uint32_t *a, uint32_t count, uint32_t base, uint32_t offset, int words) {
  uint32_t index = in_range(base, offset, count, words) ? base + offset : 0;
  uint64_t value = index < count ? a[index] : 0;
  if (words == 2 && index + 1 < count) {
    value |= uint64_t(a[index + 1]) << 32;
  }
  return value;
}

} // namespace

// Read an M0-relative source register.
// Return errors before changing outputs.
int rj_arith_s_movrels_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                           const uint32_t *a, uint32_t a_count, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  *d = uint32_t(read(a, a_count, a_base, m0, 1));
  return RJ_ARITH_SUCCESS;
}

// Read an M0-relative source register.
// Return errors before changing outputs.
int rj_arith_s_movrels_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                           const uint32_t *a, uint32_t a_count, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  *d = read(a, a_count, a_base, m0, 2);
  return RJ_ARITH_SUCCESS;
}

// Write an M0-relative destination register.
// Return errors before changing outputs.
int rj_arith_s_movreld_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t d_count, uint32_t d_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  if (in_range(d_base, m0, d_count, 1)) {
    d[d_base + m0] = a;
  }
  return RJ_ARITH_SUCCESS;
}

// Write an M0-relative destination register.
// Return errors before changing outputs.
int rj_arith_s_movreld_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                           uint32_t d_count, uint32_t d_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  if (in_range(d_base, m0, d_count, 2)) {
    d[d_base + m0] = uint32_t(a);
    d[d_base + m0 + 1] = uint32_t(a >> 32);
  }
  return RJ_ARITH_SUCCESS;
}

// Move between M0-relative registers.
// Return errors before changing outputs.
int rj_arith_s_movrelsd_2_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                              const uint32_t *a, uint32_t d_count, uint32_t a_count,
                              uint32_t d_base, uint32_t a_base, uint32_t m0) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = validate(flags, instruction_flags)) {
    return error;
  }
  uint32_t dst_offset = (m0 >> 16) & 1023;
  if (in_range(d_base, dst_offset, d_count, 1)) {
    d[d_base + dst_offset] = uint32_t(read(a, a_count, a_base, m0 & 1023, 1));
  }
  return RJ_ARITH_SUCCESS;
}
