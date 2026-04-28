// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/liveness.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <unordered_set>

namespace rocjitsu {

namespace {

void dfs_rpo(const BasicBlock &block, std::unordered_set<const BasicBlock *> &visited,
             std::vector<const BasicBlock *> &postorder) {
  if (!visited.insert(&block).second)
    return;

  for (const BasicBlock *succ : block.successors()) {
    if (succ != nullptr)
      dfs_rpo(*succ, visited, postorder);
  }
  postorder.push_back(&block);
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

void remove_vector_kills(RegisterSet &kills) {
  for (size_t i = 0; i < ISA_MAX_VGPRS; ++i)
    kills.erase({RegClass::VGPR, static_cast<uint16_t>(i), 1});
  for (size_t i = 0; i < ISA_MAX_ACC_VGPRS; ++i)
    kills.erase({RegClass::ACC_VGPR, static_cast<uint16_t>(i), 1});
}

[[nodiscard]] RegisterSet kill_defs(const InstDefUse &du) {
  RegisterSet kills = du.defs;
  if (du.has_exec_masked_vector_def)
    remove_vector_kills(kills);
  return kills;
}

} // namespace

std::vector<const BasicBlock *>
reverse_post_order(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<const BasicBlock *> postorder;
  std::unordered_set<const BasicBlock *> visited;

  for (const auto &block : blocks) {
    if (block)
      dfs_rpo(*block, visited, postorder);
  }

  std::ranges::reverse(postorder);
  return postorder;
}

LivenessAnalysis::LivenessAnalysis(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                   uint8_t wf_size) {
  liveness_.resize(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i].get(), i);
  }

  // Compute each block's local transfer function before iterating across CFG
  // edges. `gen` keeps only uses that occur before a local definition, because
  // later uses are satisfied inside the block. `kill` is every local def.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto &block = *blocks[i];
    auto &state = liveness_[i];
    for (const auto &inst : block.instructions()) {
      InstDefUse du(inst, wf_size);
      RegisterSet kills = kill_defs(du);
      RegisterSet upward_uses = du.uses;
      upward_uses -= state.kill;
      state.gen |= upward_uses;
      state.kill |= kills;
    }
  }

  const auto rpo = reverse_post_order(blocks);
  bool changed = true;
  while (changed) {
    changed = false;

    // Backward dataflow converges fastest when visiting reverse RPO.
    for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
      const BasicBlock *block = *it;
      auto idx_it = block_index_.find(block);
      if (idx_it == block_index_.end())
        continue;

      RegisterSet live_out;
      for (const BasicBlock *succ : block->successors()) {
        auto succ_idx = block_index_.find(succ);
        if (succ_idx != block_index_.end())
          live_out |= liveness_[succ_idx->second].live_in;
      }

      RegisterSet live_in = live_out;
      live_in -= liveness_[idx_it->second].kill;
      live_in |= liveness_[idx_it->second].gen;

      auto &state = liveness_[idx_it->second];
      if (state.live_out != live_out || state.live_in != live_in) {
        state.live_out = live_out;
        state.live_in = live_in;
        changed = true;
      }
    }
  }

  // Materialize live-before for instruction-level queries. The transfer
  // function is intentionally applied per instruction, so read-modify-write
  // instructions keep their source register live before the instruction even
  // when the same register is also defined by the instruction.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto &block = *blocks[i];
    RegisterSet live = liveness_[i].live_out;
    auto insts = instructions_in_order(block);
    uint64_t offset = block.start_offset();
    for (const Instruction *inst : insts) {
      instruction_by_offset_.emplace(offset, inst);
      offset += inst->size();
    }
    for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
      const Instruction *inst = *it;
      InstDefUse du(*inst, wf_size);
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

std::optional<uint16_t> LivenessAnalysis::find_free_run(uint64_t offset, uint16_t count,
                                                        uint16_t search_start) const {
  assert(count > 0 && "Must request at least one register");
  auto inst_it = instruction_by_offset_.find(offset);
  if (inst_it == instruction_by_offset_.end())
    return std::nullopt;

  const RegisterSet &live = live_before(*inst_it->second);
  for (uint16_t base = search_start; base + count <= ISA_MAX_VGPRS; ++base) {
    if (!any_live_in_range(live, RegClass::VGPR, base, count))
      return base;
  }
  return std::nullopt;
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr_pair(uint64_t offset,
                                                              uint16_t search_start) const {
  auto inst_it = instruction_by_offset_.find(offset);
  if (inst_it == instruction_by_offset_.end())
    return std::nullopt;

  const RegisterSet &live = live_before(*inst_it->second);
  constexpr uint16_t kAllocatableSgprs = 106;
  uint16_t base = search_start;
  if (base % 2 != 0)
    ++base;
  for (; base + 1 < kAllocatableSgprs; base += 2) {
    if (!any_live_in_range(live, RegClass::SGPR, base, 2))
      return base;
  }
  return std::nullopt;
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr(uint64_t offset,
                                                         uint16_t search_start) const {
  auto inst_it = instruction_by_offset_.find(offset);
  if (inst_it == instruction_by_offset_.end())
    return std::nullopt;

  const RegisterSet &live = live_before(*inst_it->second);
  constexpr uint16_t kAllocatableSgprs = 106;
  for (uint16_t base = search_start; base < kAllocatableSgprs; ++base) {
    if (!live.contains({RegClass::SGPR, base, 1}))
      return base;
  }
  return std::nullopt;
}

} // namespace rocjitsu
