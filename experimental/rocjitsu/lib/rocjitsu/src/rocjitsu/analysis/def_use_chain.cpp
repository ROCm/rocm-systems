// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <vector>

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

InstDefUse::InstDefUse(const Instruction &inst, uint8_t wf_size) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(wf_size))
      add_def(*this, *ref);
  }

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(wf_size))
      uses.expand(*ref);
  }

  std::vector<RegisterRef> implicit_defs;
  inst.implicit_defs(wf_size, implicit_defs);
  for (RegisterRef ref : implicit_defs)
    add_def(*this, ref);

  std::vector<RegisterRef> implicit_uses;
  inst.implicit_uses(wf_size, implicit_uses);
  for (RegisterRef ref : implicit_uses)
    uses.expand(ref);
}

} // namespace rocjitsu
