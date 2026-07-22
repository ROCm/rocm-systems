// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <cassert>

namespace rocjitsu {

namespace {

[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

/// @brief Distinguishes a use (may-read) expansion from a def (must-write) one.
/// @details When the VGPR-MSB bank is unknown, a USE conservatively reads any of
/// the four candidate tuples (a sound may-read over-approximation), but a DEF must
/// not claim to write all four — that would be a false must-kill of three tuples
/// the instruction does not touch. See expand_operand_register.
enum class OperandExpansionKind { Use, Def };

void expand_operand_register(RegisterSet &set, const Instruction &inst, const Operand &operand,
                             RegisterRef ref, const Gfx1250VgprMsbAnalysis *vgpr_msb,
                             OperandExpansionKind kind) {
  if (vgpr_msb == nullptr || ref.cls != RegClass::VGPR) {
    set.expand(ref);
    return;
  }

  const auto bank = vgpr_msb->bank_before(inst, operand.vgpr_msb_role());
  if (bank) {
    ref.index = static_cast<uint16_t>(ref.index + static_cast<uint16_t>(*bank) * 256u);
    set.expand(ref);
    return;
  }

  // A dynamic MODE write or disagreeing CFG predecessors can leave the bank
  // unknown. The instruction accesses exactly ONE of these four physical tuples.
  if (kind == OperandExpansionKind::Use) {
    // May-read: treating all four as possibly-read is the sound path-insensitive
    // over-approximation for liveness.
    for (uint16_t candidate = 0; candidate < 4; ++candidate) {
      RegisterRef possible = ref;
      possible.index = static_cast<uint16_t>(possible.index + candidate * 256u);
      set.expand(possible);
    }
    return;
  }

  // Must-write: expanding to all four here would falsely kill three tuples the
  // instruction does not write. Every VGPR def is exec-masked (is_exec_masked_def),
  // and kill_defs() drops VGPR kills whenever has_exec_masked_vector_def is set, so
  // an unknown-bank VGPR def never contributes a liveness kill. Record NOTHING in
  // the def set rather than an unsound must-kill; the exec-mask suppression below
  // is what keeps this sound. If that invariant ever changes (a non-exec-masked
  // VGPR def, or kill_defs no longer suppressing VGPR kills), this must be
  // revisited so an unknown-bank def does not over-kill.
  assert(is_exec_masked_def(ref) &&
         "unknown-bank VGPR def relies on exec-mask kill suppression; see kill_defs");
}

void add_def(InstDefUse &du, const Instruction &inst, const Operand &operand, RegisterRef ref,
             const Gfx1250VgprMsbAnalysis *vgpr_msb) {
  expand_operand_register(du.defs, inst, operand, ref, vgpr_msb, OperandExpansionKind::Def);
  if (is_exec_masked_def(ref))
    du.has_exec_masked_vector_def = true;
}

} // namespace

InstDefUse::InstDefUse(const Instruction &inst, const Gfx1250VgprMsbAnalysis *vgpr_msb) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      add_def(*this, inst, *op, *ref, vgpr_msb);
  }
  inst.implicit_defs(defs);

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Use);
  }
  inst.implicit_uses(uses);
}

} // namespace rocjitsu
