// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/liveness.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

void dfs_reverse_post_order(const BasicBlock &start,
                            const std::unordered_set<const BasicBlock *> &allowed,
                            std::unordered_set<const BasicBlock *> &visited,
                            std::vector<const BasicBlock *> &postorder) {
  if (!allowed.contains(&start) || !visited.insert(&start).second)
    return;

  std::vector<std::pair<const BasicBlock *, size_t>> stack;
  stack.emplace_back(&start, 0);

  while (!stack.empty()) {
    auto &[block, next_successor] = stack.back();
    const auto &successors = block->successors();
    if (next_successor < successors.size()) {
      const BasicBlock *succ = successors[next_successor++];
      if (succ != nullptr && allowed.contains(succ) && visited.insert(succ).second)
        stack.emplace_back(succ, 0);
      continue;
    }

    postorder.push_back(block);
    stack.pop_back();
  }
}

std::vector<const Instruction *> instructions_in_order(BasicBlock &block) {
  std::vector<const Instruction *> insts;
  for (const auto &inst : block.instructions())
    insts.push_back(&inst);
  return insts;
}

[[nodiscard]] bool any_live_in_range(const RegisterSet &live, RegClass cls, uint16_t base,
                                     uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (live.contains({cls, static_cast<uint16_t>(base + i), 1}))
      return true;
  }
  return false;
}

[[nodiscard]] RegisterSet kill_defs(const InstDefUse &du) {
  RegisterSet kills = du.defs;
  // Predicated defs and EXEC-masked vector defs preserve old values on at least
  // one path or lane. Until EXEC state is tracked at each program point, those
  // writes cannot be treated as unconditional liveness kills.
  if (du.has_predicated_def)
    return {};
  if (du.has_exec_masked_vector_def) {
    kills.clear_class(RegClass::VGPR);
    kills.clear_class(RegClass::ACC_VGPR);
  }
  return kills;
}

} // namespace

std::vector<const BasicBlock *> reverse_post_order(KernelBlockScope blocks) {
  std::vector<const BasicBlock *> postorder;
  std::unordered_set<const BasicBlock *> allowed;
  std::unordered_set<const BasicBlock *> visited;

  allowed.reserve(blocks.size());
  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      allowed.insert(block);
  }

  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      dfs_reverse_post_order(*block, allowed, visited, postorder);
  }

  std::ranges::reverse(postorder);
  return postorder;
}

LivenessAnalysis::LivenessAnalysis(KernelBlockScope blocks, LivenessAnalysisOptions options) {
  min_free_vgpr_ = options.min_free_vgpr;
  analyze(blocks);
}

void LivenessAnalysis::analyze(KernelBlockScope blocks) {
  liveness_.resize(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i], i);
  }

  // Compute each block's local transfer function before iterating across CFG
  // edges. `gen` keeps only uses that occur before a local definition, because
  // later uses are satisfied inside the block. `kill` is every local def.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto *block = blocks[i];
    if (block == nullptr)
      continue;
    auto &state = liveness_[i];
    for (const auto &inst : block->instructions()) {
      InstDefUse du(inst);
      RegisterSet kills = kill_defs(du);
      RegisterSet upward_uses = du.uses;
      upward_uses -= state.kill;
      state.gen |= upward_uses;
      state.kill |= kills;
    }
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
    auto idx_it = block_index_.find(block);
    if (idx_it != block_index_.end())
      enqueue(idx_it->second);
  }

  while (!worklist.empty()) {
    const size_t idx = worklist.front();
    worklist.pop_front();
    in_worklist[idx] = false;

    const BasicBlock *block = blocks[idx];
    if (block == nullptr)
      continue;

    RegisterSet live_out;
    for (const BasicBlock *succ : block->successors()) {
      auto succ_idx = block_index_.find(succ);
      if (succ_idx != block_index_.end())
        live_out |= liveness_[succ_idx->second].live_in;
    }

    RegisterSet live_in = live_out;
    live_in -= liveness_[idx].kill;
    live_in |= liveness_[idx].gen;

    auto &state = liveness_[idx];
    const bool live_in_changed = state.live_in != live_in;
    if (state.live_out != live_out || live_in_changed) {
      state.live_out = live_out;
      state.live_in = live_in;

      if (live_in_changed) {
        for (const BasicBlock *pred : block->predecessors()) {
          auto pred_idx = block_index_.find(pred);
          if (pred_idx != block_index_.end())
            enqueue(pred_idx->second);
        }
      }
    }
  }

  // Materialize live-before for instruction-level queries. The transfer
  // function is intentionally applied per instruction, so read-modify-write
  // instructions keep their source register live before the instruction even
  // when the same register is also defined by the instruction.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto *block = blocks[i];
    if (block == nullptr)
      continue;
    RegisterSet live = liveness_[i].live_out;
    auto insts = instructions_in_order(*block);
    for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
      const Instruction *inst = *it;
      InstDefUse du(*inst);
      RegisterSet kills = kill_defs(du);
      live -= kills;
      live |= du.uses;
      live_before_.emplace(inst, live);
    }
  }
}

const BlockLiveness &LivenessAnalysis::block_liveness(const BasicBlock &block) const {
  auto it = block_index_.find(&block);
  if (it == block_index_.end())
    throw std::out_of_range("block_liveness: block was not part of this analysis");
  return liveness_.at(it->second);
}

const RegisterSet &LivenessAnalysis::live_before(const Instruction &inst) const {
  auto it = live_before_.find(&inst);
  return it != live_before_.end() ? it->second : empty_;
}

bool LivenessAnalysis::is_live_before(const Instruction &inst, RegisterRef ref) const {
  return live_before(inst).contains(ref);
}

std::optional<uint16_t> LivenessAnalysis::find_free_run(const Instruction *inst, uint16_t count,
                                                        uint16_t search_start) const {
  assert(count > 0 && "Must request at least one register");
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end())
    return std::nullopt;

  const RegisterSet &live = live_it->second;
  const size_t first_candidate = std::max<size_t>(search_start, min_free_vgpr_);
  for (size_t base = first_candidate; base + count <= REGISTER_SET_MAX_VGPRS; ++base) {
    if (!any_live_in_range(live, RegClass::VGPR, static_cast<uint16_t>(base), count))
      return static_cast<uint16_t>(base);
  }
  return std::nullopt;
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr_pair(const Instruction *inst,
                                                              uint16_t search_start) const {
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end())
    return std::nullopt;

  const RegisterSet &live = live_it->second;
  size_t base = search_start;
  if (base % 2 != 0)
    ++base; // even-align for s_mov_b64-style pair moves.
  for (; base + 1 < REGISTER_SET_ALLOCATABLE_SGPRS; base += 2) {
    if (!any_live_in_range(live, RegClass::SGPR, static_cast<uint16_t>(base), 2))
      return static_cast<uint16_t>(base);
  }
  return std::nullopt;
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr(const Instruction *inst,
                                                         uint16_t search_start) const {
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end())
    return std::nullopt;

  const RegisterSet &live = live_it->second;
  // Keep this in sync with find_free_sgpr_pair(): only normal SGPRs that are
  // valid across supported families are candidates for temporary allocation.
  for (size_t base = search_start; base < REGISTER_SET_ALLOCATABLE_SGPRS; ++base) {
    if (!live.contains({RegClass::SGPR, static_cast<uint16_t>(base), 1}))
      return static_cast<uint16_t>(base);
  }
  return std::nullopt;
}

} // namespace rocjitsu
