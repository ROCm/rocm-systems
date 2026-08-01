// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace rocjitsu {

namespace {

[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

[[nodiscard]] bool is_vop3_carry_out(std::string_view mnemonic) {
  return mnemonic == "v_add_co_u32" || mnemonic == "v_add_co_ci_u32" ||
         mnemonic == "v_sub_co_u32" || mnemonic == "v_sub_co_ci_u32" ||
         mnemonic == "v_subrev_co_u32" || mnemonic == "v_subrev_co_ci_u32" ||
         mnemonic == "v_mad_co_u64_u32" || mnemonic == "v_mad_co_i64_i32";
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
  // kill three the instruction does not write, so record NOTHING in the def set.
  // This is only sound because such a def never contributes a liveness kill:
  // control reaches here only for a VGPR ref (see the early return above), every
  // VGPR def is exec-masked (is_exec_masked_def), and kill_defs() drops all VGPR
  // kills once has_exec_masked_vector_def is set. If any of those change (a
  // non-exec-masked VGPR def, or kill_defs no longer suppressing VGPR kills), an
  // unknown-bank def would start over-killing and this must record the precise
  // physical tuple instead. (An assert(is_exec_masked_def(ref)) here would be
  // tautological — ref is already known to be a VGPR — so the invariant is
  // documented rather than checked.)
}

void add_def(InstDefUse &du, const Instruction &inst, const Operand &operand, RegisterRef ref,
             const Gfx1250VgprMsbAnalysis *vgpr_msb, UnknownVgprDefPolicy unknown_vgpr_defs) {
  expand_operand_register(du.defs, inst, operand, ref, vgpr_msb, OperandExpansionKind::Def,
                          unknown_vgpr_defs);
  if (is_exec_masked_def(ref))
    du.has_exec_masked_vector_def = true;
}

} // namespace

std::optional<RegisterRef> wave_mode_destination_ref(const Instruction &inst,
                                                     const Operand &operand, int operand_index,
                                                     uint32_t wavefront_size) {
  auto ref = operand.to_register_ref();
  if (!ref || ref->cls != RegClass::SGPR)
    return std::nullopt;

  const std::string_view mnemonic = inst.mnemonic();
  const bool is_vopc_mask = operand_index == 0 && mnemonic.starts_with("v_cmp");
  const bool is_carry_mask = operand_index == 1 && is_vop3_carry_out(mnemonic);
  const bool is_div_scale_mask = operand_index == 1 && mnemonic.starts_with("v_div_scale_");
  if (!is_vopc_mask && !is_carry_mask && !is_div_scale_mask)
    return std::nullopt;

  ref->width = wavefront_size == 32 ? 1 : 2;
  return ref;
}

InstDefUse::InstDefUse(const Instruction &inst, const Gfx1250VgprMsbAnalysis *vgpr_msb,
                       UnknownVgprDefPolicy unknown_vgpr_defs) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      add_def(*this, inst, *op, *ref, vgpr_msb, unknown_vgpr_defs);
  }
  // No generated instruction currently reports an implicit VGPR def. If one is
  // added for gfx1250, it must expose an operand with a VGPR-MSB role so global
  // usage can resolve the physical bank instead of recording only the raw low
  // eight-bit index.
  inst.implicit_defs(defs);

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Use,
                              unknown_vgpr_defs);
  }

  if (vgpr_msb == nullptr) {
    // No dynamic VGPR banking (non-gfx1250): the flat hook already reports every
    // implicit read at its physical index.
    inst.implicit_uses(uses);
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
    if (auto ref = op->to_register_ref())
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb, OperandExpansionKind::Use,
                              unknown_vgpr_defs);
  }

  // The flat hook also reports encoded-field implicit reads with no backing
  // operand (e.g. FLAT/GLOBAL saddr, an SGPR). Merge only those: the VGPR reads
  // it would add carry no bank, and the operand path above already resolved them
  // to the correct physical tuple, so re-adding the raw low-8 index would mark a
  // wrong, unbanked register live.
  RegisterSet flat_implicit;
  inst.implicit_uses(flat_implicit);
  flat_implicit.clear_class(RegClass::VGPR);
  uses |= flat_implicit;
}

} // namespace rocjitsu
