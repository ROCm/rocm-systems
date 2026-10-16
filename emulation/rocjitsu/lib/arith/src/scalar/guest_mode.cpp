// SPDX-License-Identifier: MIT

// Explicit updates of guest MODE rounding and denormal-control fields.
// Owns public entry points and their validation/dispatch policy.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Update selected guest MODE fields. Reject invalid controls before changing outputs.
template <unsigned Shift>
int set_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate, uint32_t *mode) {
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  if (mode) {
    // This writes the supplied guest register only, never the host FP control state.
    *mode = (*mode & ~(15U << Shift)) | (uint32_t(immediate & 15) << Shift);
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Update guest MODE rounding controls.
// Return errors before changing outputs.
int rj_arith_s_round_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                          uint32_t *mode) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return set_mode<0>(flags, instruction_flags, immediate, mode);
}

// Update guest MODE denormal controls.
// Return errors before changing outputs.
int rj_arith_s_denorm_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                           uint32_t *mode) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return set_mode<4>(flags, instruction_flags, immediate, mode);
}
