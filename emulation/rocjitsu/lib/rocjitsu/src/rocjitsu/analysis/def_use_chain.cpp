// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {

namespace {

// A vector def (VGPR/AccVGPR) normally preserves inactive lanes under EXEC, so
// it cannot always be treated as an unconditional kill. The exception is instructions
// flagged IGNORES_EXEC (e.g. branch-class ops). We flag non-EXEC ignoring vector
// def instructions as "EXEC masked def" so once the EXEC state at that instruction is
// determined, liveness analysis can properly assign the vector def as full kill or not.
[[nodiscard]] bool is_vector_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

} // namespace

InstDefUse::InstDefUse(const Instruction &inst) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;
  const bool ignores_exec = inst.flags() & IGNORES_EXEC;
  bool has_vector_def = false;

  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref()) {
      defs.expand(*ref);
      has_vector_def |= is_vector_def(*ref);
    }
  }
  has_exec_masked_vector_def = has_vector_def && !ignores_exec;
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
