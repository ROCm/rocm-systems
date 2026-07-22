// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/exec_state.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rocjitsu {

namespace {

/// @brief How an instruction affects the EXEC mask.
enum class ExecWrite : uint8_t {
  None,      ///< Does not write EXEC; state unchanged.
  AllOnes,   ///< Writes the whole EXEC mask to all-ones -> Full.
  Preserve,  ///< Sets a subset of EXEC to all-ones (e.g. only exec_lo on
             ///< Wave64); keeps an already-Full mask Full and leaves Unknown
             ///< as Unknown, but cannot establish Full on its own.
  Narrowing, ///< Writes EXEC some other way -> Unknown.
};

[[nodiscard]] bool writes_exec(const Instruction &inst) {
  // Two complementary signals: the WRITES_EXEC flag covers instructions whose
  // semantics always write EXEC (s_*_saveexec, s_wrexec, v_cmpx), while an EXEC
  // destination operand covers the generic move case (`s_mov_b64 exec, ...`),
  // which has no flag because writing EXEC is a property of the instance's
  // destination, not the opcode.
  if (inst.flags() & WRITES_EXEC)
    return true;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    auto ref = op->to_register_ref();
    if (ref && ref->cls == RegClass::EXEC)
      return true;
  }
  return false;
}

/// @brief All-ones mask for an EXEC register of @p wave_size lanes.
[[nodiscard]] uint64_t full_exec_mask(uint32_t wave_size) {
  return wave_size >= 64 ? ~0ULL : ((1ULL << wave_size) - 1ULL);
}

/// @brief The EXEC bits an instruction writes: a mask of the written bits and
/// whether they cover the whole @p wave_size-lane EXEC register.
///
/// @details Semantic EXEC writers (WRITES_EXEC: saveexec/wrexec/v_cmpx) write the
/// whole mask. A generic write via an EXEC destination operand covers only that
/// operand's lanes — e.g. `s_mov_b32 exec_lo` writes 32 bits, which is the whole
/// mask on Wave32 but only half on Wave64.
struct ExecWriteExtent {
  uint64_t mask = 0; ///< Bits written into EXEC.
  bool full = false; ///< True when those bits cover the entire EXEC register.
};

[[nodiscard]] ExecWriteExtent exec_write_extent(const Instruction &inst, uint32_t wave_size) {
  if (inst.flags() & WRITES_EXEC)
    return {full_exec_mask(wave_size), true};
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::EXEC) {
      const int w = op->size_bits();
      const uint64_t mask = (w >= 64) ? ~0ULL : ((1ULL << w) - 1ULL);
      return {mask, w >= static_cast<int>(wave_size)};
    }
  }
  return {};
}

/// @brief True when @p op is a compile-time all-ones constant across @p mask.
[[nodiscard]] bool src_const_is_all_ones(const Operand *op, uint64_t mask) {
  if (op == nullptr)
    return false;
  const std::optional<uint64_t> cv = op->const_value();
  return cv && (*cv & mask) == mask;
}

/// @brief How the value written to the destination is formed from the sources.
enum class Combinator { Other, Copy, Or };

[[nodiscard]] Combinator combinator(const Instruction &inst) {
  if (inst.flags() & RESULT_OR)
    return Combinator::Or;
  if (inst.flags() & RESULT_COPY)
    return Combinator::Copy;
  return Combinator::Other;
}

/// @brief True when the value written into the EXEC bits is provably all-ones
/// across @p mask (the written width).
///
/// @details Operation-aware via the result combinator and compile-time-constant
/// sources (`const_value()` resolves literals and inline constants without a
/// wavefront):
///   * `Copy` (`exec = src`):     the single source is all-ones.
///   * `Or`   (`exec = a | b …`): any source is an all-ones constant.
///   * everything else (and/xor/not/cmpx/register restores/...): not provable.
[[nodiscard]] bool writes_all_ones_value(const Instruction &inst, uint64_t mask) {
  switch (combinator(inst)) {
  case Combinator::Copy:
    return inst.num_src_operands() == 1 && src_const_is_all_ones(inst.src_operand(0), mask);
  case Combinator::Or:
    for (int i = 0; i < inst.num_src_operands(); ++i)
      if (src_const_is_all_ones(inst.src_operand(i), mask))
        return true;
    return false;
  case Combinator::Other:
    return false;
  }
  return false;
}

/// @details A full all-ones write establishes `Full`. A *partial* all-ones write
/// (e.g. `s_mov_b32 exec_lo, -1` on Wave64) only sets a subset of the mask to
/// ones, so it keeps an already-`Full` mask `Full` but cannot establish `Full`
/// from `Unknown` — that is `Preserve`. AND-style writes (incl.
/// `s_and_saveexec exec, -1`, where `exec & -1 == exec`) and writes of any other
/// value fall through to `Narrowing`.
[[nodiscard]] ExecWrite classify(const Instruction &inst, uint32_t wave_size) {
  if (!writes_exec(inst))
    return ExecWrite::None;
  const ExecWriteExtent ext = exec_write_extent(inst, wave_size);
  if (writes_all_ones_value(inst, ext.mask))
    return ext.full ? ExecWrite::AllOnes : ExecWrite::Preserve;
  return ExecWrite::Narrowing;
}

[[nodiscard]] ExecState transfer(ExecState in, const Instruction &inst, uint32_t wave_size) {
  switch (classify(inst, wave_size)) {
  case ExecWrite::AllOnes:
    return ExecState::Full;
  case ExecWrite::Narrowing:
    return ExecState::Unknown;
  case ExecWrite::Preserve: // partial all-ones: keep Full as Full, Unknown as Unknown
  case ExecWrite::None:
    break;
  }
  return in;
}

/// @brief Lattice meet: `Full` only when both inputs are `Full`.
[[nodiscard]] ExecState meet(ExecState a, ExecState b) {
  return (a == ExecState::Full && b == ExecState::Full) ? ExecState::Full : ExecState::Unknown;
}

[[nodiscard]] ExecState block_transfer(ExecState in, BasicBlock &block, uint32_t wave_size) {
  ExecState state = in;
  for (const auto &inst : block.instructions())
    state = transfer(state, inst, wave_size);
  return state;
}

} // namespace

ExecMaskAnalysis::ExecMaskAnalysis(KernelBlockScope blocks, uint8_t wave_size,
                                   std::span<const ScopedCfgEdge> extra_edges)
    : wave_size_(wave_size) {
  if (wave_size != 32 && wave_size != 64)
    throw std::invalid_argument("ExecMaskAnalysis: wave_size must be 32 or 64");
  analyze(blocks, extra_edges);
}

void ExecMaskAnalysis::analyze(KernelBlockScope blocks,
                               std::span<const ScopedCfgEdge> extra_edges) {
  states_.assign(blocks.size(), BlockExec{});
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i], i);
  }

  // BasicBlock::successors()/predecessors() carry only context-free local CFG
  // edges. Fold the caller-provided scoped call/return edges into an index-based
  // adjacency the same way LivenessAnalysis does, so both analyses see the same
  // graph. Only edges whose endpoints are both in scope are kept.
  std::vector<std::vector<size_t>> successors(blocks.size());
  std::vector<std::vector<size_t>> predecessors(blocks.size());
  auto add_edge = [&](const BasicBlock *from, const BasicBlock *to) {
    auto from_it = block_index_.find(from);
    auto to_it = block_index_.find(to);
    if (from_it == block_index_.end() || to_it == block_index_.end())
      return;
    auto &succs = successors[from_it->second];
    if (std::ranges::find(succs, to_it->second) != succs.end())
      return;
    succs.push_back(to_it->second);
    predecessors[to_it->second].push_back(from_it->second);
  };

  for (const BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const BasicBlock *succ : block->successors())
      add_edge(block, succ);
  }
  for (const ScopedCfgEdge &edge : extra_edges)
    add_edge(edge.from, edge.to);

  // A block is an entry when no in-scope predecessor reaches it (scoped edges
  // included). Entries are pinned to `Unknown`; interior blocks start
  // optimistically `Full` so the forward `must` meet can pull them down to
  // `Unknown` to a fixpoint.
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] == nullptr)
      continue;
    // consider entry to be scope leader and blocks with no predecessors in scope
    states_[i].is_entry = (predecessors[i].empty() || i == 0);
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

  // Seed in reverse-post-order for fast forward convergence, then queue any
  // remaining blocks: a block reachable only through a scoped edge is absent
  // from the successors()-based RPO but still needs to be processed.
  for (const BasicBlock *block : rpo) {
    auto it = block_index_.find(block);
    if (it != block_index_.end())
      enqueue(it->second);
  }
  for (size_t idx = 0; idx < blocks.size(); ++idx)
    enqueue(idx);

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
      for (size_t pred_idx : predecessors[idx]) {
        const ExecState pred_out = states_[pred_idx].out;
        acc = acc ? meet(*acc, pred_out) : pred_out;
      }
      in = acc.value_or(ExecState::Unknown);
    }

    const ExecState out = block_transfer(in, *block, wave_size_);
    if (in != states_[idx].in || out != states_[idx].out) {
      states_[idx].in = in;
      states_[idx].out = out;
      for (size_t succ_idx : successors[idx])
        enqueue(succ_idx);
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
      state = transfer(state, inst, wave_size_);
    }
  }
}

ExecState ExecMaskAnalysis::before(const Instruction &inst) const {
  auto it = before_.find(&inst);
  return it != before_.end() ? it->second : ExecState::Unknown;
}

} // namespace rocjitsu
