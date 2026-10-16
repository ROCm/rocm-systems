// SPDX-License-Identifier: MIT

// Guest hardware-register bitfield reads/writes and unsupported-state handling.
// Owns public entry points and their validation/dispatch policy.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Extract the offset/width-selected field encoded in hwreg from a register value.
uint32_t extract_hwreg(uint32_t value, uint16_t hwreg) {
  unsigned offset = (hwreg >> 6) & 31;
  unsigned width = (hwreg >> 11) + 1;
  return (value >> offset) & uint32_t((1ULL << width) - 1);
}

} // namespace

// Read the selected guest hardware-register field.
// Return errors before changing outputs.
int rj_arith_s_getreg_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t hwreg,
                          uint32_t mode, uint32_t status, uint32_t trapsts, uint32_t gpr_alloc,
                          uint32_t lds_alloc, uint32_t ib_sts, uint64_t pc, uint32_t ib_dbg1,
                          uint32_t sh_mem_bases, uint64_t shader_tba, uint64_t perf_snapshot_pc,
                          uint64_t flat_scratch, uint32_t hw_id1, uint32_t hw_id2,
                          uint32_t pops_packer, uint32_t sched_mode, uint32_t perf_snapshot_data,
                          uint32_t ib_sts2, uint64_t shader_cycles) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  uint32_t value;
  switch (hwreg & 63) {
  case 1:
    value = mode;
    break;
  case 2:
    value = status;
    break;
  case 3:
    value = trapsts;
    break;
  case 5:
    value = gpr_alloc;
    break;
  case 6:
    value = lds_alloc;
    break;
  case 7:
    value = ib_sts;
    break;
  case 8:
    value = uint32_t(pc);
    break;
  case 9:
    value = uint32_t(pc >> 32);
    break;
  case 13:
    value = ib_dbg1;
    break;
  case 15:
    value = sh_mem_bases;
    break;
  case 16:
    value = uint32_t(shader_tba);
    break;
  case 17:
    value = uint32_t(shader_tba >> 32);
    break;
  case 18:
    value = uint32_t(perf_snapshot_pc);
    break;
  case 19:
    value = uint32_t(perf_snapshot_pc >> 32);
    break;
  case 20:
    value = uint32_t(flat_scratch);
    break;
  case 21:
    value = uint32_t(flat_scratch >> 32);
    break;
  case 23:
    value = hw_id1;
    break;
  case 24:
    value = hw_id2;
    break;
  case 25:
    value = pops_packer;
    break;
  case 26:
    value = sched_mode;
    break;
  case 27:
    value = perf_snapshot_data;
    break;
  case 28:
    value = ib_sts2;
    break;
  case 29:
    value = uint32_t(shader_cycles);
    break;
  default:
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  *d = extract_hwreg(value, hwreg);
  return RJ_ARITH_SUCCESS;
}

// Read the selected guest hardware-register field.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_getreg_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                uint16_t hwreg, uint32_t mode, uint32_t status, uint32_t state_priv,
                                uint32_t gpr_alloc, uint32_t lds_alloc,
                                uint32_t perf_snapshot_data_in, uint32_t perf_snapshot_pc_lo_in,
                                uint32_t perf_snapshot_pc_hi_in, uint32_t perf_snapshot_data1_in,
                                uint32_t perf_snapshot_data2_in, uint32_t excp_flag_priv,
                                uint32_t excp_flag_user, uint32_t trap_ctrl, uint64_t scratch_base,
                                uint32_t hw_id1, uint32_t hw_id2, uint32_t ib_sts2,
                                uint64_t shader_cycles, uint32_t *perf_snapshot_data,
                                uint32_t *perf_snapshot_pc_lo, uint32_t *perf_snapshot_pc_hi,
                                uint32_t *perf_snapshot_data1, uint32_t *perf_snapshot_data2) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // PC_HI reads unlock the snapshot. Its precise state update is not modeled.
  if ((hwreg & 63) == 12 && (flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_LOOSE &&
      (perf_snapshot_data || perf_snapshot_pc_lo || perf_snapshot_pc_hi || perf_snapshot_data1 ||
       perf_snapshot_data2)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }
  uint32_t value;
  switch (hwreg & 63) {
  case 1:
    value = mode;
    break;
  case 2:
    value = status;
    break;
  case 4:
    value = state_priv;
    break;
  case 5:
    value = gpr_alloc;
    break;
  case 6:
    value = lds_alloc;
    break;
  case 10:
    value = perf_snapshot_data_in;
    break;
  case 11:
    value = perf_snapshot_pc_lo_in;
    break;
  case 12:
    value = perf_snapshot_pc_hi_in;
    break;
  case 15:
    value = perf_snapshot_data1_in;
    break;
  case 16:
    value = perf_snapshot_data2_in;
    break;
  case 17:
    value = excp_flag_priv;
    break;
  case 18:
    value = excp_flag_user;
    break;
  case 19:
    value = trap_ctrl;
    break;
  case 20:
    value = uint32_t(scratch_base);
    break;
  case 21:
    value = uint32_t(scratch_base >> 32);
    break;
  case 23:
    value = hw_id1;
    break;
  case 24:
    value = hw_id2;
    break;
  case 28:
    value = ib_sts2;
    break;
  case 29:
    value = uint32_t(shader_cycles);
    break;
  case 30:
    value = uint32_t(shader_cycles >> 32);
    break;
  default:
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  *d = extract_hwreg(value, hwreg);
  return RJ_ARITH_SUCCESS;
}

// Write the selected guest hardware-register field.
// Return errors before changing outputs.
int rj_arith_s_setreg_b32(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg, uint32_t a,
                          uint32_t status_in, uint32_t *mode, uint32_t *status, uint32_t *trapsts,
                          uint32_t *flush_ib, uint64_t *flat_scratch) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  unsigned id = hwreg & 63;
  bool privileged = status_in & (1U << 5);
  if ((flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_LOOSE &&
      ((id == 14 && flush_ib) ||
       ((id == 20 || id == 21) && privileged && (flat_scratch || status)))) {
    return RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }

  unsigned offset = (hwreg >> 6) & 31;
  unsigned width = (hwreg >> 11) + 1;
  uint32_t field_mask = uint32_t(((1ULL << width) - 1) << offset);
  uint32_t *destination = nullptr;
  uint32_t writable = 0;
  switch (id) {
  case 1:
    destination = mode;

    // FP/DX10/IEEE/LOD/trap controls [11:0], EXCP_EN [19:12,21], FP16_OVFL
    // [23], DISABLE_PERF [27]. Exception-enable index 8 (MODE bit 20) is reserved.
    writable = 0x08afffff;
    break;
  case 2:
    destination = status;

    // Table 5's privileged-writable SCC/priorities, EXPORT_RDY, HALT,
    // SKIP_EXPORT, CDBG, LDS_PARAM_RDY, MUST_EXPORT and SCRATCH_EN fields.
    writable = privileged ? 0x2a34211f : 0;
    break;
  case 3:
    destination = trapsts;

    // EXCP [8:0], SAVECTX/ILLEGAL/ADDR_WATCH/BUFFER_OOB/HOST_TRAP/
    // WAVE_START/WAVE_END [18:10], TRAP_AFTER_INST [20].
    writable = 0x0017fdff;
    break;
  case 14:
  case 20:
  case 21:
    // FLUSH_IB and scratch effects are not guessed; null opts out and loose
    // semantics skip unavailable reporting. Unprivileged scratch writes are no-ops.
    return RJ_ARITH_SUCCESS;

  // Allocation/status, PC, SH_MEM_BASES, TBA, snapshots, IDs and the clock.
  // TBA is read-only in the ISA manual; rocjitsu's privileged descriptor differs.
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  case 13:
  case 15:
  case 16:
  case 17:
  case 18:
  case 19:
  case 23:
  case 24:
  case 25:
  case 26:
  case 27:
  case 28:
  case 29:
    return RJ_ARITH_SUCCESS;
  default:
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  uint32_t mask = field_mask & writable;
  if (mask && destination) {
    *destination = (*destination & ~mask) | ((a << offset) & mask);
  }
  return RJ_ARITH_SUCCESS;
}

// Write the selected guest hardware-register field.
// Return errors before changing outputs.
int rj_arith_s_setreg_imm32_b32(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                uint32_t immediate, uint32_t status_in, uint32_t *mode,
                                uint32_t *status, uint32_t *trapsts, uint32_t *flush_ib,
                                uint64_t *flat_scratch) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_setreg_b32(flags, instruction_flags, hwreg, immediate, status_in, mode, status,
                               trapsts, flush_ib, flat_scratch);
}

// Write the selected guest hardware-register field.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_setreg_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                uint32_t a, uint32_t status, uint32_t *mode, uint32_t *state_priv,
                                uint32_t *excp_flag_priv, uint32_t *excp_flag_user,
                                uint32_t *trap_ctrl, uint64_t *scratch_base) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  unsigned id = hwreg & 63;
  bool privileged = status & (1U << 5);

  // The alignment requirement does not establish the scratch write mask or
  // whether writing it affects STATE_PRIV.SCRATCH_EN. Do not invent updates.
  if ((id == 20 || id == 21) && privileged && (scratch_base || state_priv) &&
      (flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (int error = rocjitsu::arith::validate(flags, uint32_t(instruction_flags), true)) {
    return error;
  }

  unsigned offset = (hwreg >> 6) & 31;
  unsigned width = (hwreg >> 11) + 1;

  // A 64-bit intermediate handles width 32 and clips fields crossing bit 31.
  uint32_t field_mask = uint32_t(((1ULL << width) - 1) << offset);
  uint32_t *destination = nullptr;
  uint32_t writable = 0;
  switch (id) {
  case 1:
    destination = mode;

    // FP modes [7:0], FP16_OVFL [23], scalar prefetch [24], DISABLE_PERF [27].
    writable = 0x098000ff;
    break;
  case 4:
    destination = state_priv;

    // WG/sleep/barrier [2:0], SCC/priorities/HALT [14:9], SCRATCH_EN [18].
    writable = privileged ? 0x00047e07 : 0;
    break;
  case 17:
    destination = excp_flag_priv;

    // Exception/status flags [11:0], first memory-violation source [31:30].
    writable = privileged ? 0xc0000fff : 0;
    break;
  case 18:
    destination = excp_flag_user;

    // Arithmetic [6:0], buffer OOB [30], LOD clamped [31].
    writable = 0xc000007f;
    break;
  case 19:
    destination = trap_ctrl;

    // Arithmetic, address-watch, wave-end and trap-after-instruction enables.
    writable = privileged ? 0x000003ff : 0;
    break;
  case 20:
  case 21:
    // Null opts out; loose mode skips unavailable scratch-state reporting.
    return RJ_ARITH_SUCCESS;

  // STATUS, allocation state, snapshots, hardware IDs, IB_STS2 and clock halves.
  case 2:
  case 5:
  case 6:
  case 10:
  case 11:
  case 12:
  case 15:
  case 16:
  case 23:
  case 24:
  case 28:
  case 29:
  case 30:
    return RJ_ARITH_SUCCESS;
  default:
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  uint32_t mask = field_mask & writable;
  if (mask && destination) {
    *destination = (*destination & ~mask) | ((a << offset) & mask);
  }
  return RJ_ARITH_SUCCESS;
}

// Write the selected guest hardware-register field.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_setreg_imm32_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                      uint32_t immediate, uint32_t status, uint32_t *mode,
                                      uint32_t *state_priv, uint32_t *excp_flag_priv,
                                      uint32_t *excp_flag_user, uint32_t *trap_ctrl,
                                      uint64_t *scratch_base) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_s_setreg_b32_gfx12(flags, instruction_flags, hwreg, immediate, status, mode,
                                     state_priv, excp_flag_priv, excp_flag_user, trap_ctrl,
                                     scratch_base);
}
