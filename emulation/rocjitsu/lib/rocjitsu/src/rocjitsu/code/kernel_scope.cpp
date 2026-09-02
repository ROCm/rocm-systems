// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_scope.h"

#include "rocjitsu/code/basic_block.h"

#include <algorithm>
#include <cassert>
#include <functional>

namespace rocjitsu {

BlockOffsetIndex build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockOffsetIndex index;
  index.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      index.emplace_back(block->start_offset(), block.get());
  }
  std::ranges::sort(index, {}, &std::pair<uint64_t, BasicBlock *>::first);
  return index;
}

BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockPositionIndex index;
  index.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      index.emplace(blocks[i].get(), i);
  }
  return index;
}

BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset) {
  auto it = std::ranges::upper_bound(index, offset, std::less<>{},
                                     &std::pair<uint64_t, BasicBlock *>::first);
  if (it == index.begin())
    return nullptr;
  --it;

  BasicBlock *block = it->second;
  if (block == nullptr || offset >= block->end_offset())
    return nullptr;
  return block;
}

std::vector<BasicBlock *> reachable_kernel_blocks(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, const BlockOffsetIndex &block_index,
    const BlockPositionIndex &block_positions, BasicBlock &entry, const KernelScopeSpec &spec) {
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<size_t> reached_indices;
  std::vector<size_t> stack;
  auto push_block = [&](BasicBlock *block) {
    auto it = block_positions.find(block);
    if (it != block_positions.end())
      stack.push_back(it->second);
  };
  push_block(&entry);
  for (const uint64_t own_entry : spec.own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      push_block(extra_entry);
    }
  }

  while (!stack.empty()) {
    const size_t block_idx = stack.back();
    stack.pop_back();
    if (block_idx >= blocks.size() || reachable[block_idx])
      continue;
    reachable[block_idx] = 1;
    reached_indices.push_back(block_idx);
    BasicBlock *block = blocks[block_idx].get();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!spec.own_entries.contains(succ->start_offset()) &&
          spec.kernel_entries.contains(succ->start_offset()))
        continue;
      push_block(succ);
    }
    // Ordinary CFG successors describe control that always follows from the
    // current program counter: fallthroughs, conditional targets, direct branch
    // targets, and recovered non-returning setpc targets. Call edges are tracked
    // separately because a shared callee block can return to different
    // continuations depending on which call site entered it. Reachability for
    // translation still has to include the callee body, but later liveness gets
    // explicit call/return edges rather than treating every possible return as a
    // global CFG successor.
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      // A body whose address is taken is emitted exactly once, as an adopted root, and every
      // pointer to it names that one copy. Cloning it into each caller's scope would put the same
      // source offset at several placements and leave relocate_relative_text_addends() choosing
      // between them -- which is the reasoning kernel_translation_scopes() already documents for
      // adopted roots, applied here to the callees a relocation-table dispatch reaches.
      //
      // Not doing so is a divergence, not merely waste: the addend is rewritten to the canonical
      // clone, so a later translation gives every dispatch site a call edge to a clone owned by
      // some other scope, which that scope then clones again. Each pass adds another copy.
      //
      // Only the indirect edges may be dropped. A direct s_call_b64 to the same body leaves a
      // BranchFixup naming that source offset in this scope, and patch_direct_branch_fixups()
      // resolves it against the kernel-local layout, so dropping the body makes that call
      // unresolvable. Nothing rewrites a direct call through the addend, so a local clone cannot
      // create the placement ambiguity the indirect case has.
      const bool address_taken_indirect_callee =
          call.kind == BasicBlock::CallEdgeKind::IndirectSwapPc &&
          spec.address_taken_entries.contains(callee->start_offset());
      if (!spec.own_entries.contains(callee->start_offset()) &&
          (spec.kernel_entries.contains(callee->start_offset()) || address_taken_indirect_callee))
        continue;
      push_block(callee);
    }
  }

  std::ranges::sort(reached_indices);
  std::vector<BasicBlock *> ordered;
  ordered.reserve(reached_indices.size());
  for (size_t block_idx : reached_indices) {
    if (blocks[block_idx])
      ordered.push_back(blocks[block_idx].get());
  }
  return ordered;
}

} // namespace rocjitsu
