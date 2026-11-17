// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file control_flow.h
/// @brief Shared instruction-level control-flow predicates.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/instruction.h"

#include <cstdint>

namespace rocjitsu {

/// @brief Whether an instruction ends the current program path.
///
/// @details Trap ID 2 is an abort only when the caller establishes a terminating
/// handler contract. Offline analysis cannot infer that contract from the ISA.
[[nodiscard]] inline bool is_software_trap_two(const Instruction &inst, rj_code_arch_t arch) {
  if ((inst.flags() & SOFTWARE_TRAP) == 0 || inst.size() != sizeof(uint32_t))
    return false;
  const uint32_t *raw = inst.raw_encoding();
  const uint32_t trap_id_mask =
      (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) ? 0xfu : 0xffu;
  return raw != nullptr && (raw[0] & trap_id_mask) == 2;
}

/// @param assume_trap_two_aborts Preserve the established ROCr abort contract
/// for existing callers; offline checkers pass false when a handler may be absent.
[[nodiscard]] inline bool is_program_path_terminator(const Instruction &inst,
                                                     rj_code_arch_t arch,
                                                     bool assume_trap_two_aborts = true) {
  if (is_software_trap_two(inst, arch))
    return assume_trap_two_aborts;
  if ((inst.flags() & PROGRAM_TERMINATOR) != 0)
    return true;
  return false;
}

/// @brief Whether an indirect branch can also continue at the next instruction.
///
/// @details Like s_setpc, S_CBRANCH_G_FORK and S_CBRANCH_JOIN take a successor
/// from SGPRs, but they also fall through: a fork whose fail side runs first, or
/// a join on its second arrival. Callers still treat the SGPR-held successor as
/// unknown.
[[nodiscard]] inline bool indirect_branch_falls_through(const Instruction &inst) {
  constexpr uint64_t kFlags = INDIRECT_BRANCH | BRANCH_STACK;
  return (inst.flags() & kFlags) == kFlags;
}

/// @brief Whether an instruction can transfer control away from fallthrough.
///
/// @details Some system instructions express their PC write only as an operand,
/// without a branch flag. Traps transfer to a handler even when they have no
/// explicit PC destination. Check path termination separately.
[[nodiscard]] inline bool is_control_flow_transfer(const Instruction &inst) {
  constexpr uint64_t kTransfers =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | SOFTWARE_TRAP;
  if ((inst.flags() & kTransfers) != 0)
    return true;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *operand = inst.dst_operand(i);
    if (operand && operand->to_special_reg_class() == RegClass::PC)
      return true;
  }
  return false;
}

} // namespace rocjitsu
