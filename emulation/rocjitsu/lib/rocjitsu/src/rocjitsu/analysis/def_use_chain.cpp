// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {

namespace {

[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

void expand_operand_register(RegisterSet &set, const Instruction &inst, const Operand &operand,
                             RegisterRef ref, const Gfx1250VgprMsbAnalysis *vgpr_msb) {
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
  // unknown. The instruction accesses one of these four physical tuples, but
  // treating all four as possible is the sound path-insensitive approximation
  // for liveness and scratch-clobber checks.
  for (uint16_t candidate = 0; candidate < 4; ++candidate) {
    RegisterRef possible = ref;
    possible.index = static_cast<uint16_t>(possible.index + candidate * 256u);
    set.expand(possible);
  }
}

void add_def(InstDefUse &du, const Instruction &inst, const Operand &operand, RegisterRef ref,
             const Gfx1250VgprMsbAnalysis *vgpr_msb) {
  expand_operand_register(du.defs, inst, operand, ref, vgpr_msb);
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
      expand_operand_register(uses, inst, *op, *ref, vgpr_msb);
  }
  inst.implicit_uses(uses);
}

} // namespace rocjitsu
