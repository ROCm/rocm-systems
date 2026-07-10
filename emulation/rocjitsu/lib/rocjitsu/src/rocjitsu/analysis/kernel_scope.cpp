// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/kernel_scope.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <unordered_set>

namespace rocjitsu {

namespace {

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        const BlockOffsetIndex &block_index, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        const std::unordered_set<uint64_t> &own_entries) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<BasicBlock *> stack{&entry};
  for (const uint64_t own_entry : own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      stack.push_back(extra_entry);
    }
  }

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");
    if (!reachable.insert(block).second)
      continue;

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!own_entries.contains(succ->start_offset()) &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back(succ);
    }
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      if (!own_entries.contains(callee->start_offset()) &&
          kernel_entries.contains(callee->start_offset()))
        continue;
      stack.push_back(callee);
    }
  }

  std::vector<BasicBlock *> ordered;
  ordered.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      ordered.push_back(block.get());
  }
  return ordered;
}

[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  if (inst.size() != sizeof(uint32_t) || inst.mnemonic() != "s_setpc_b64")
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  std::vector<BasicBlock *> returns;
  std::vector<BasicBlock *> stack{&callee};
  std::unordered_set<BasicBlock *> visited;

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    assert(block != nullptr && "return-block walk stack should contain only decoded blocks");
    if (!allowed_blocks.contains(block) || !visited.insert(block).second)
      continue;

    const Instruction *term = block->terminator();
    assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");
    if (s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), return_sreg)) {
      returns.push_back(block);
      continue;
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back(succ);
    }
  }

  return returns;
}

void add_scoped_call_flow(KernelCfgScope &scope, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(scope.blocks.size());
  for (BasicBlock *block : scope.blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  for (BasicBlock *block : scope.blocks) {
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      scope.liveness_edges.push_back({.from = block, .to = call.callee});
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        scope.liveness_edges.push_back({.from = return_block, .to = call.continuation});
        const Instruction *term = return_block->terminator();
        assert(term != nullptr && "return block should contain a terminator");
        scope.call_return_offsets.insert(term->src_loc());
      }
    }
  }
}

} // namespace

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

std::optional<KernelCfgScope>
build_kernel_cfg_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                       const BlockOffsetIndex &block_index, const KernelScopeRequest &request,
                       std::span<const uint64_t> all_kernel_entries,
                       std::span<const uint8_t> text) {
  BasicBlock *entry = block_for_offset(block_index, request.entry_offset);
  if (entry == nullptr)
    return std::nullopt;

  std::unordered_set<uint64_t> own_entries{request.entry_offset};
  for (const uint64_t extra : request.additional_entry_offsets) {
    if (block_for_offset(block_index, extra) == nullptr)
      return std::nullopt;
    own_entries.insert(extra);
  }

  std::unordered_set<uint64_t> kernel_entries(all_kernel_entries.begin(), all_kernel_entries.end());
  KernelCfgScope scope;
  scope.entry = entry;
  scope.blocks = reachable_kernel_blocks(blocks, block_index, *entry, kernel_entries, own_entries);
  add_scoped_call_flow(scope, text);
  return scope;
}

} // namespace rocjitsu
