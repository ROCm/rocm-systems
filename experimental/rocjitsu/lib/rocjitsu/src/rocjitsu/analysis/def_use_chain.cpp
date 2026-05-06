// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {

namespace {

// TODO: Replace this class-based approximation with instruction metadata that
// identifies vector defs whose inactive lanes are preserved under EXEC, and pair
// it with program-point EXEC state so full-EXEC writes can be treated as normal
// kills.
[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

void add_def(InstDefUse &du, RegisterRef ref) {
  du.defs.expand(ref);
  if (is_exec_masked_def(ref))
    du.has_exec_masked_vector_def = true;
}

} // namespace

InstDefUse::InstDefUse(const Instruction &inst) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      add_def(*this, *ref);
  }
  inst.implicit_defs(defs);

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      uses.expand(*ref);
  }
  inst.implicit_uses(uses);
}

} // namespace rocjitsu
