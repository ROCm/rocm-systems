// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/def_use_chain.h"

#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {

namespace {

// Checks if a RegisterRef class is a vector (VGPR/AccVGPR).
[[nodiscard]] bool is_vector_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

// InstDefUse surfaces only ordinary register-file refs (SGPR/VGPR/AccVGPR)
// through to_register_ref(). A generic selector operand can decode to a special
// register (e.g. EXEC_LO/EXEC_HI in an OPR_SDST field), but that ref collapses
// the LO/HI halves into one class-wide singleton, so it is dropped here rather
// than recorded imprecisely. Genuinely
// fieldless special operands are surfaced instead through to_special_reg_class().
// See def_use_chain.h.
[[nodiscard]] std::optional<RegisterRef> ordinary_register_ref(const Operand &op) {
  auto ref = op.to_register_ref();
  if (ref && is_special_reg_class(ref->cls))
    return std::nullopt;
  return ref;
}

/// @brief Distinguishes a use (may-read) expansion from a def (must-write) one.
/// @details When the VGPR-MSB bank is unknown, a USE conservatively reads any of
/// the four candidate tuples (a sound may-read over-approximation), but a DEF must
/// not claim to write all four — that would be a false must-kill of three tuples
/// the instruction does not touch. See expand_operand_register.
enum class OperandExpansionKind { Use, Def };

void expand_operand_register(RegisterSet &set, const Instruction &inst, const Operand &operand,
                             RegisterRef ref, const Gfx1250VgprMsbAnalysis *vgpr_msb,
                             OperandExpansionKind kind, UnknownVgprDefPolicy unknown_vgpr_defs) {
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
  // unknown. The instruction accesses exactly ONE of these four physical
  // tuples. A may-read, and a must-write recorded for whole-kernel usage rather
  // than for kills, both need the sound over-approximation.
  if (kind == OperandExpansionKind::Use || unknown_vgpr_defs == UnknownVgprDefPolicy::ExpandAll) {
    for (uint16_t candidate = 0; candidate < 4; ++candidate) {
      RegisterRef possible = ref;
      possible.index = static_cast<uint16_t>(possible.index + candidate * 256u);
      set.expand(possible);
    }
    return;
  }

  // Must-write with an unknown bank: expanding to all four tuples would falsely
  // kill three tuples the instruction does not write (an over-kill, which is
  // unsound for liveness), so record NOTHING in the def set instead. An
  // unrecorded def can only cause an under-kill, which merely keeps a value live
  // longer than necessary and is always a sound (conservative) approximation.
  // Known-bank defs above record the precise physical tuple and kill as usual,
  // subject to the EXEC-masking rules in kill_defs().
}

} // namespace

InstDefUse::InstDefUse(const Instruction &inst, const Gfx1250VgprMsbAnalysis *vgpr_msb,
                       UnknownVgprDefPolicy unknown_vgpr_defs) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;
  const bool ignores_exec = inst.flags() & IGNORES_EXEC;
  bool has_vector_def = false;

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = ordinary_register_ref(*op)) {
      expand_operand_register(defs, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Def,
                              unknown_vgpr_defs);
      has_vector_def |= is_vector_def(*ref);
    } else if (auto sc = op->to_special_reg_class()) {
      defs.expand(RegisterRef{*sc, 0, 1}); // singleton special def
    }
  }
  has_exec_masked_vector_def = has_vector_def && !ignores_exec;
  // No generated instruction currently reports an implicit VGPR def. If one is
  // added for gfx1250, it must expose an operand with a VGPR-MSB role so global
  // usage can resolve the physical bank instead of recording only the raw low
  // eight-bit index. Merge only the ordinary projection: an implicit hook that
  // surfaces a special via a generic selector (to_register_ref()) collapses it
  // to a class-wide singleton, so it is dropped here for the same reason the
  // explicit dst loop above drops selector-encoded specials.
  RegisterSet implicit_defs_set;
  inst.implicit_defs(implicit_defs_set);
  defs |= implicit_defs_set.ordinary_only();

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = ordinary_register_ref(*op))
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Use,
                              unknown_vgpr_defs);
    else if (auto sc = op->to_special_reg_class())
      uses.expand(RegisterRef{*sc, 0, 1}); // singleton special use
  }

  if (vgpr_msb == nullptr) {
    // No dynamic VGPR banking (non-gfx1250): the flat hook already reports every
    // implicit read at its physical index. Merge only the ordinary projection so
    // a selector-encoded special surfaced by the hook (e.g. an exec_lo
    // preserve-read on a partial-write op) is dropped rather than recorded as an
    // imprecise class-wide singleton — matching the explicit src loop above.
    RegisterSet implicit_uses_set;
    inst.implicit_uses(implicit_uses_set);
    uses |= implicit_uses_set.ordinary_only();
    return;
  }

  // On gfx1250 an implicit read with a backing operand (a partial-write/RMW op
  // preserve-reading its destination, or a swap preserve-reading both operands)
  // carries its own VGPR-MSB role and width, so resolve it through the same
  // per-operand path as explicit sources. This applies each read's own bank —
  // critical when a preserve-read aliases an explicit bank-0 source but sits in a
  // different destination bank, or when a swap mixes SRC0 and DST banks — and it
  // preserves the operand's true tuple width.
  std::vector<const Operand *> implicit_use_operand_list;
  inst.implicit_use_operands(implicit_use_operand_list);
  for (const Operand *op : implicit_use_operand_list) {
    if (op == nullptr)
      continue;
    if (auto ref = ordinary_register_ref(*op))
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Use,
                              unknown_vgpr_defs);
  }

  // The flat hook also reports encoded-field implicit reads with no backing
  // operand (e.g. FLAT/GLOBAL saddr, an SGPR). Merge only those: the VGPR reads
  // it would add carry no bank, and the operand path above already resolved them
  // to the correct physical tuple, so re-adding the raw low-8 index would mark a
  // wrong, unbanked register live. Also drop any selector-encoded special the
  // hook surfaces, for the same reason as the non-gfx1250 path above.
  RegisterSet flat_implicit;
  inst.implicit_uses(flat_implicit);
  flat_implicit.clear_class(RegClass::VGPR);
  uses |= flat_implicit.ordinary_only();
}

} // namespace rocjitsu
