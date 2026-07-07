// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {

namespace {

[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

// A 32-bit register lane is the finest granularity RegisterSet tracks. An
// operand narrower than that (e.g. a 16-bit dst) writes only part of its
// lowest lane and leaves the remaining bits intact, so the instruction
// read-modify-writes that register rather than fully redefining it.
[[nodiscard]] bool is_partial_def(int size_bits) {
  return size_bits > 0 && size_bits < REGISTER_GRANULARITY;
}

void add_def(InstDefUse &du, RegisterRef ref, int size_bits) {
  du.defs.expand(ref);
  // Record a sub-lane def as a use as well: liveness must keep the register
  // live across the instruction because the untouched bits survive the write.
  if (is_partial_def(size_bits))
    du.uses.expand(ref);
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
      add_def(*this, *ref, op->size_bits());
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
