// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/exec_state.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <stdexcept>
#include <unordered_set>
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

[[nodiscard]] bool writes_execution_mask(const Instruction &inst) {
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
/// operand's lanes at the sub-register's position: `exec_lo` (index 0) writes
/// bits [0,32), `exec_hi` (index 1) bits [32,64). Only an exec_lo-based write can
/// cover the active mask; an exec_hi write never touches lanes [0,wave_size).
struct ExecWriteExtent {
  uint64_t mask = 0;     ///< Bits written into EXEC (sub-register width, unpositioned).
  bool full = false;     ///< True when those bits cover the entire active EXEC mask.
  bool disjoint = false; ///< True when the written bits lie entirely outside the active
                         ///< mask [0,wave_size) (e.g. exec_hi on Wave32), so the write
                         ///< touches no active lane whatever value it stores.
};

[[nodiscard]] ExecWriteExtent exec_write_extent(const Instruction &inst, uint32_t wave_size) {
  if (inst.flags() & WRITES_EXEC)
    return {full_exec_mask(wave_size), /*full=*/true, /*disjoint=*/false};
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::EXEC) {
      const int w = op->size_bits();
      const uint64_t mask = (w >= 64) ? ~0ULL : ((1ULL << w) - 1ULL);
      // Full only when the write starts at exec_lo (index 0) and spans the mask.
      const bool full = ref->index == 0 && w >= static_cast<int>(wave_size);
      // Position the written bits within EXEC (exec_lo -> [0,w), exec_hi ->
      // [32,32+w)) and test them against the active mask. A write landing wholly
      // above the active lanes -- most notably exec_hi on Wave32 -- leaves every
      // active lane untouched.
      const int shift = ref->index * 32;
      const uint64_t positioned = shift >= 64 ? 0ULL : (mask << shift);
      const bool disjoint = (positioned & full_exec_mask(wave_size)) == 0;
      return {mask, full, disjoint};
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

/// @details A write disjoint from the active mask (e.g. `s_mov_b32 exec_hi, N` on
/// Wave32, where the active lanes are exec_lo) touches no active lane, so it is
/// `Preserve` regardless of the value written. Otherwise: a full all-ones write
/// establishes `Full`; a *partial* all-ones write (e.g. `s_mov_b32 exec_lo, -1`
/// on Wave64) only sets a subset of the mask to ones, so it keeps an already-`Full`
/// mask `Full` but cannot establish `Full` from `Unknown` — that is `Preserve`.
/// AND-style writes (incl. `s_and_saveexec exec, -1`, where `exec & -1 == exec`)
/// and writes of any other value fall through to `Narrowing`.
[[nodiscard]] ExecWrite classify(const Instruction &inst, uint32_t wave_size) {
  if (!writes_execution_mask(inst))
    return ExecWrite::None;
  const ExecWriteExtent ext = exec_write_extent(inst, wave_size);
  if (ext.disjoint)
    return ExecWrite::Preserve;
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

/// @brief A block's whole transfer function over the two-point lattice.
///
/// @details Since every per-instruction transfer is either a constant
/// (AllOnes -> Full, Narrowing -> Unknown) or the identity (Preserve/None), the
/// composition over a block is fully determined by its *last* constant EXEC
/// write. Summarizing it once lets the worklist apply a block in O(1) instead of
/// re-walking (and re-classifying) the whole instruction list on every re-visit.
enum class BlockEffect : uint8_t {
  Identity,     ///< No constant EXEC write: out == in.
  ConstFull,    ///< Last constant write is all-ones: out == Full.
  ConstUnknown, ///< Last constant write narrows EXEC: out == Unknown.
};

[[nodiscard]] BlockEffect summarize_block_exec_effect(BasicBlock &block, uint32_t wave_size) {
  BlockEffect effect = BlockEffect::Identity;
  for (const auto &inst : block.instructions()) {
    switch (classify(inst, wave_size)) {
    case ExecWrite::AllOnes:
      effect = BlockEffect::ConstFull;
      break;
    case ExecWrite::Narrowing:
      effect = BlockEffect::ConstUnknown;
      break;
    case ExecWrite::Preserve:
    case ExecWrite::None:
      break;
    }
  }
  return effect;
}

/// @brief Summarize every in-scope block's transfer function once, up front, so
/// the fixpoint worklist can apply each block in O(1) without re-walking (and
/// re-classifying) its instructions on every re-visit.
[[nodiscard]] std::vector<BlockEffect> summarize_all_block_exec_effects(KernelBlockScope blocks,
                                                                        uint32_t wave_size) {
  std::vector<BlockEffect> effects(blocks.size(), BlockEffect::Identity);
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      effects[i] = summarize_block_exec_effect(*blocks[i], wave_size);
  }
  return effects;
}

[[nodiscard]] ExecState apply_block_effect(BlockEffect effect, ExecState in) {
  switch (effect) {
  case BlockEffect::ConstFull:
    return ExecState::Full;
  case BlockEffect::ConstUnknown:
    return ExecState::Unknown;
  case BlockEffect::Identity:
    break;
  }
  return in;
}

} // namespace

ExecMaskAnalysis::ExecMaskAnalysis(KernelBlockScope blocks, uint8_t wave_size,
                                   std::span<const ScopedCfgEdge> extra_edges,
                                   std::span<const BasicBlock *const> entry_blocks)
    : wave_size_(wave_size) {
  if (wave_size != 32 && wave_size != 64)
    throw std::invalid_argument("ExecMaskAnalysis: wave_size must be 32 or 64");
  analyze(blocks, extra_edges, entry_blocks);
}

void ExecMaskAnalysis::analyze(KernelBlockScope blocks, std::span<const ScopedCfgEdge> extra_edges,
                               std::span<const BasicBlock *const> entry_blocks) {
  states_.assign(blocks.size(), BlockExec{});
  block_index_.reserve(blocks.size());
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

  // A block is an entry when it is a caller-supplied kernel entry or has no
  // in-scope predecessor (scoped edges included). Entries are pinned to
  // `Unknown`; interior blocks start optimistically `Full` so the forward `must`
  // meet can pull them down to `Unknown` to a fixpoint. A real entry may be a
  // loop header with a backedge, so predecessor count alone cannot find it --
  // hence the explicit entry_blocks. Over-marking a block as entry only loses
  // precision, never soundness.
  std::unordered_set<const BasicBlock *> entry_set(entry_blocks.begin(), entry_blocks.end());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] == nullptr)
      continue;
    states_[i].is_entry = predecessors[i].empty() || entry_set.contains(blocks[i]);
  }
  // Fallback when no entry was supplied and none was found (e.g. a fully cyclic
  // scope): seed the scope leader so the meet has at least one `Unknown` source.
  if (entry_blocks.empty() &&
      std::ranges::none_of(states_, [](const BlockExec &s) { return s.is_entry; })) {
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i] != nullptr) {
        states_[i].is_entry = true;
        break;
      }
    }
  }

  const std::vector<BlockEffect> block_effects =
      summarize_all_block_exec_effects(blocks, wave_size_);

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

    const ExecState out = apply_block_effect(block_effects[idx], in);
    if (in != states_[idx].in || out != states_[idx].out) {
      states_[idx].in = in;
      states_[idx].out = out;
      for (size_t succ_idx : successors[idx])
        enqueue(succ_idx);
    }
  }

  // Materialize the EXEC state entering each instruction. Size full_before_ once
  // from the total instruction count so the per-instruction inserts never
  // rehash.
  size_t total_instructions = 0;
  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      total_instructions += block->num_instructions();
  }
  full_before_.reserve(total_instructions);
  for (size_t i = 0; i < blocks.size(); ++i) {
    BasicBlock *block = blocks[i];
    if (block == nullptr)
      continue;
    ExecState state = states_[i].in;
    for (const auto &inst : block->instructions()) {
      if (state == ExecState::Full)
        full_before_.insert(&inst);
      state = transfer(state, inst, wave_size_);
    }
  }
}

ExecState ExecMaskAnalysis::before(const Instruction &inst) const {
  return full_before_.contains(&inst) ? ExecState::Full : ExecState::Unknown;
}

} // namespace rocjitsu
