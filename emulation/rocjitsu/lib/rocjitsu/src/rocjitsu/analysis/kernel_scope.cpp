// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/kernel_scope.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu {

namespace {

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

struct ReachableKernelBlocks {
  std::vector<BasicBlock *> blocks;
  std::unordered_map<const BasicBlock *, KernelCfgOwnerProofKind> proofs;
};

[[nodiscard]] KernelCfgOwnerProofKind call_proof(KernelCfgOwnerProofKind caller,
                                                 BasicBlock::CallEdgeKind edge) {
  const KernelCfgOwnerProofKind call = edge == BasicBlock::CallEdgeKind::DirectCall
                                           ? KernelCfgOwnerProofKind::DirectCall
                                           : KernelCfgOwnerProofKind::RecoveredIndirectCall;
  return static_cast<uint8_t>(caller) >= static_cast<uint8_t>(call) ? caller : call;
}

[[nodiscard]] ReachableKernelBlocks
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        const BlockOffsetIndex &block_index, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        const std::unordered_set<uint64_t> &own_entries) {
  struct WorkItem {
    BasicBlock *block = nullptr;
    KernelCfgOwnerProofKind proof = KernelCfgOwnerProofKind::KernelLocal;
  };
  std::unordered_map<const BasicBlock *, KernelCfgOwnerProofKind> reachable;
  std::vector<WorkItem> stack{{.block = &entry}};
  for (const uint64_t own_entry : own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      stack.push_back({.block = extra_entry});
    }
  }

  while (!stack.empty()) {
    const WorkItem item = stack.back();
    stack.pop_back();
    BasicBlock *block = item.block;
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");
    const auto prior = reachable.find(block);
    if (prior != reachable.end() &&
        static_cast<uint8_t>(prior->second) <= static_cast<uint8_t>(item.proof))
      continue;
    reachable[block] = item.proof;

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!own_entries.contains(succ->start_offset()) &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back({.block = succ, .proof = item.proof});
    }
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      if (!own_entries.contains(callee->start_offset()) &&
          kernel_entries.contains(callee->start_offset()))
        continue;
      stack.push_back({.block = callee, .proof = call_proof(item.proof, call.kind)});
    }
  }

  ReachableKernelBlocks result;
  result.blocks.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      result.blocks.push_back(block.get());
  }
  result.proofs = std::move(reachable);
  return result;
}

[[nodiscard]] std::optional<uint16_t> s_setpc_sreg(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t) ||
      (inst.mnemonic() != "s_setpc_b64" && inst.mnemonic() != "s_set_pc_i64"))
    return std::nullopt;
  return static_cast<uint16_t>(word & 0xffu);
}

struct FunctionReturnWalk {
  std::vector<BasicBlock *> matching_returns;
  std::vector<BasicBlock *> encountered_setpcs;
};

[[nodiscard]] FunctionReturnWalk
walk_function_returns(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                      const std::unordered_set<BasicBlock *> &allowed_blocks) {
  FunctionReturnWalk result;
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
    if (const auto setpc_sreg = s_setpc_sreg(*term, text_word_at(text, term->src_loc()))) {
      result.encountered_setpcs.push_back(block);
      if (*setpc_sreg == return_sreg) {
        result.matching_returns.push_back(block);
        continue;
      }
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back(succ);
    }
  }

  return result;
}

void add_scoped_call_flow(KernelCfgScope &scope, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(scope.blocks.size());
  for (BasicBlock *block : scope.blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::unordered_map<BasicBlock *, std::unordered_set<uint16_t>> reaching_return_sregs;
  std::unordered_set<BasicBlock *> matching_return_blocks;
  for (BasicBlock *block : scope.blocks) {
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee))
        continue;

      scope.liveness_edges.push_back({.from = block, .to = call.callee});
      const FunctionReturnWalk return_walk =
          walk_function_returns(*call.callee, call.return_sreg, text, allowed_blocks);
      for (BasicBlock *return_block : return_walk.encountered_setpcs)
        reaching_return_sregs[return_block].insert(call.return_sreg);
      // Even when the continuation belongs to another kernel scope, the call
      // still reaches the shared helper and contributes a return-register
      // context. It cannot add a scoped return edge or prove a return block
      // exempt from ordinary indirect control flow.
      if (!allowed_blocks.contains(call.continuation))
        continue;
      for (BasicBlock *return_block : return_walk.matching_returns) {
        scope.liveness_edges.push_back({.from = return_block, .to = call.continuation});
        matching_return_blocks.insert(return_block);
      }
    }
  }

  for (BasicBlock *return_block : matching_return_blocks) {
    const Instruction *term = return_block->terminator();
    assert(term != nullptr && "return block should contain a terminator");
    const auto owner_proof = scope.owner_proofs.find(return_block);
    const auto contexts = reaching_return_sregs.find(return_block);
    const auto return_sreg = s_setpc_sreg(*term, text_word_at(text, term->src_loc()));
    // A syntactic return is safe to exempt from ordinary indirect-branch
    // handling only when every path to its block entered through a validated
    // call and every such call expects the exact pair consumed by this setpc.
    // Shared helper code reached from calls with different return registers is
    // therefore left as ordinary indirect control flow.
    if (owner_proof != scope.owner_proofs.end() &&
        owner_proof->second != KernelCfgOwnerProofKind::KernelLocal &&
        contexts != reaching_return_sregs.end() && contexts->second.size() == 1u && return_sreg &&
        contexts->second.contains(*return_sreg)) {
      scope.call_return_offsets.insert(term->src_loc());
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

std::vector<KernelCfgOwnerProof>
kernel_cfg_owners_for_offset(std::span<const KernelCfgOwnerScope> owner_scopes,
                             const BlockOffsetIndex &block_index, uint64_t offset) {
  BasicBlock *block = block_for_offset(block_index, offset);
  if (block == nullptr)
    return {};
  std::vector<KernelCfgOwnerProof> owners;
  for (const KernelCfgOwnerScope &owner : owner_scopes) {
    const auto proof = owner.scope.owner_proofs.find(block);
    if (proof != owner.scope.owner_proofs.end())
      owners.push_back(
          {.descriptor_file_offset = owner.descriptor_file_offset, .kind = proof->second});
  }
  return owners;
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
  ReachableKernelBlocks reachable =
      reachable_kernel_blocks(blocks, block_index, *entry, kernel_entries, own_entries);
  scope.blocks = std::move(reachable.blocks);
  scope.owner_proofs = std::move(reachable.proofs);
  add_scoped_call_flow(scope, text);
  return scope;
}

} // namespace rocjitsu
