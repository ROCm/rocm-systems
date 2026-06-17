// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/exec_state.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <deque>
#include <optional>

namespace rocjitsu {

namespace {

/// @brief How an instruction affects the EXEC mask.
enum class ExecWrite {
  None,      ///< Does not write EXEC.
  Narrowing, ///< Writes EXEC, but not a provable all-ones value.
  AllOnes,   ///< Provably writes an all-ones EXEC mask.
};

[[nodiscard]] bool writes_exec(const Instruction &inst) {
  if (inst.flags() & WRITES_EXEC)
    return true;
  // Generated to_register_ref() may not surface EXEC yet; the flag is the
  // primary signal. This is the fallback for operands that do resolve to EXEC.
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::EXEC)
      return true;
  }
  return false;
}

/// @brief Bit mask covering the EXEC write width (defaults to 64-bit/Wave64
/// when no EXEC destination operand exposes its width).
[[nodiscard]] uint64_t exec_width_mask(const Instruction &inst) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::EXEC) {
      const int w = op->size_bits();
      if (w > 0 && w < 64)
        return (1ULL << w) - 1ULL;
      return ~0ULL;
    }
  }
  return ~0ULL;
}

/// @brief True when the instruction provably assigns an all-ones EXEC mask.
///
/// @details Conservative: the single source must expose an all-ones compile-time
/// value through Operand::const_value(), which resolves both literals
/// (`s_mov_b64 exec, 0xffffffffffffffff`) and inline constants
/// (`s_mov_b64 exec, -1`) without any wavefront. Restores from a register,
/// save/restore, AND/OR/XOR narrowing, and v_cmpx all fail this test (their
/// source is not a compile-time constant) and therefore yield `Unknown`.
/// Missing a real all-ones write only costs precision (the point stays
/// `Unknown`); it is never unsound.
[[nodiscard]] bool writes_all_ones(const Instruction &inst) {
  if (inst.num_src_operands() != 1)
    return false;
  const Operand *src = inst.src_operand(0);
  if (src == nullptr)
    return false;
  const std::optional<uint64_t> cv = src->const_value();
  if (!cv)
    return false;
  const uint64_t mask = exec_width_mask(inst);
  return (*cv & mask) == mask;
}

[[nodiscard]] ExecWrite classify(const Instruction &inst) {
  if (!writes_exec(inst))
    return ExecWrite::None;
  return writes_all_ones(inst) ? ExecWrite::AllOnes : ExecWrite::Narrowing;
}

[[nodiscard]] ExecState transfer(ExecState in, const Instruction &inst) {
  switch (classify(inst)) {
  case ExecWrite::AllOnes:
    return ExecState::Full;
  case ExecWrite::Narrowing:
    return ExecState::Unknown;
  case ExecWrite::None:
    break;
  }
  return in;
}

/// @brief Lattice meet: `Full` only when both inputs are `Full`.
[[nodiscard]] ExecState meet(ExecState a, ExecState b) {
  return (a == ExecState::Full && b == ExecState::Full) ? ExecState::Full : ExecState::Unknown;
}

[[nodiscard]] ExecState block_transfer(ExecState in, BasicBlock &block) {
  ExecState state = in;
  for (const auto &inst : block.instructions())
    state = transfer(state, inst);
  return state;
}

} // namespace

ExecMaskAnalysis::ExecMaskAnalysis(KernelBlockScope blocks) { analyze(blocks); }

void ExecMaskAnalysis::analyze(KernelBlockScope blocks) {
  states_.assign(blocks.size(), BlockExec{});
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i], i);
  }

  // A block is an entry when no predecessor is part of this scope. Entries are
  // pinned to `Unknown`; interior blocks start optimistically `Full` so the
  // forward `must` meet can pull them down to `Unknown` to a fixpoint.
  for (size_t i = 0; i < blocks.size(); ++i) {
    const BasicBlock *block = blocks[i];
    if (block == nullptr)
      continue;
    bool has_in_scope_pred = false;
    for (const BasicBlock *pred : block->predecessors()) {
      if (block_index_.contains(pred)) {
        has_in_scope_pred = true;
        break;
      }
    }
    states_[i].is_entry = !has_in_scope_pred;
  }

  const auto rpo = reverse_post_order(blocks);
  std::deque<size_t> worklist;
  std::vector<bool> in_worklist(blocks.size(), false);
  auto enqueue = [&](size_t idx) {
    if (idx >= in_worklist.size() || in_worklist[idx])
      return;
    in_worklist[idx] = true;
    worklist.push_back(idx);
  };

  for (const BasicBlock *block : rpo) {
    auto it = block_index_.find(block);
    if (it != block_index_.end())
      enqueue(it->second);
  }

  while (!worklist.empty()) {
    const size_t idx = worklist.front();
    worklist.pop_front();
    in_worklist[idx] = false;

    BasicBlock *block = blocks[idx];
    if (block == nullptr)
      continue;

    ExecState in;
    if (states_[idx].is_entry) {
      in = ExecState::Unknown;
    } else {
      std::optional<ExecState> acc;
      for (const BasicBlock *pred : block->predecessors()) {
        auto pred_it = block_index_.find(pred);
        if (pred_it == block_index_.end())
          continue;
        const ExecState pred_out = states_[pred_it->second].out;
        acc = acc ? meet(*acc, pred_out) : pred_out;
      }
      in = acc.value_or(ExecState::Unknown);
    }

    const ExecState out = block_transfer(in, *block);
    if (in != states_[idx].in || out != states_[idx].out) {
      states_[idx].in = in;
      states_[idx].out = out;
      for (const BasicBlock *succ : block->successors()) {
        auto succ_it = block_index_.find(succ);
        if (succ_it != block_index_.end())
          enqueue(succ_it->second);
      }
    }
  }

  // Materialize the EXEC state entering each instruction.
  for (size_t i = 0; i < blocks.size(); ++i) {
    BasicBlock *block = blocks[i];
    if (block == nullptr)
      continue;
    ExecState state = states_[i].in;
    for (const auto &inst : block->instructions()) {
      before_.emplace(&inst, state);
      state = transfer(state, inst);
    }
  }
}

ExecState ExecMaskAnalysis::before(const Instruction &inst) const {
  auto it = before_.find(&inst);
  return it != before_.end() ? it->second : ExecState::Unknown;
}

} // namespace rocjitsu
