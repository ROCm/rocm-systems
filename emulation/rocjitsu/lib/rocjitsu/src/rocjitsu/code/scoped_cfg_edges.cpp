// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/scoped_cfg_edges.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <cassert>
#include <cstring>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

namespace rocjitsu {

namespace {

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

/// @brief True when @p inst is exactly `s_setpc_b64/s_set_pc_i64 s[ssrc0:+1]`.
///
/// @details Return-like scalar control flow is left as an indirect branch in the
/// translated stream, so callers must validate that the terminator reads the
/// call edge's saved return SGPR. Checks the raw SOP1 source field rather than
/// broader semantics so only the exact scoped call-return form matches.
[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  const std::string_view mnemonic = inst.mnemonic();
  if (inst.size() != sizeof(uint32_t) || (mnemonic != "s_setpc_b64" && mnemonic != "s_set_pc_i64"))
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

/// @brief Find return blocks inside one context-sensitive call target.
///
/// @details The same helper block can be entered by multiple call sites, and the
/// correct continuation is selected by the return SGPR written at that site. The
/// walk stays inside @p allowed_blocks: nested callees are visited with their own
/// return SGPR as a stopping condition while their continuations retain the
/// enclosing condition, exposing paths that return directly through an enclosing
/// pair without mistaking a nested callee's return for the enclosing one.
[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  struct WalkPoint {
    BasicBlock *block = nullptr;
    std::optional<uint16_t> terminal_return_sreg;
  };

  std::vector<BasicBlock *> returns;
  std::unordered_set<BasicBlock *> return_set;
  std::vector<WalkPoint> stack{{.block = &callee, .terminal_return_sreg = std::nullopt}};
  std::set<std::pair<BasicBlock *, std::optional<uint16_t>>> visited;

  while (!stack.empty()) {
    const WalkPoint point = stack.back();
    stack.pop_back();
    BasicBlock *block = point.block;
    assert(block != nullptr && "return-block walk stack should contain only decoded blocks");
    if (!allowed_blocks.contains(block) ||
        !visited.insert({block, point.terminal_return_sreg}).second)
      continue;

    const Instruction *term = block->terminator();
    // BasicBlock::build() never emits an empty block, so this is an invariant rather than an
    // expected input -- but the assert compiles out, and a null here would be dereferenced while
    // loading a code object. Skipping costs a return-edge classification; crashing costs the host
    // process. A block with no instructions also has no terminator to classify, so there is
    // nothing this walk could have concluded from it anyway.
    assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");
    if (term == nullptr)
      continue;
    if (point.terminal_return_sreg && s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()),
                                                        *point.terminal_return_sreg)) {
      continue;
    }
    if (s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), return_sreg)) {
      if (return_set.insert(block).second)
        returns.push_back(block);
      continue;
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back({.block = succ, .terminal_return_sreg = point.terminal_return_sreg});
    }
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      stack.push_back({.block = call.callee, .terminal_return_sreg = call.return_sreg});
      stack.push_back(
          {.block = call.continuation, .terminal_return_sreg = point.terminal_return_sreg});
    }
  }

  return returns;
}

} // namespace

std::vector<ScopedCfgEdge> scoped_call_liveness_edges(std::span<BasicBlock *const> blocks,
                                                      std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::vector<ScopedCfgEdge> edges;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      edges.push_back({.from = block, .to = call.callee});
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        edges.push_back({.from = return_block, .to = call.continuation});
      }
    }
  }

  return edges;
}

std::unordered_set<uint64_t> scoped_call_return_offsets(std::span<BasicBlock *const> blocks,
                                                        std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::unordered_set<uint64_t> returns;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        const Instruction *term = return_block->terminator();
        assert(term != nullptr && "function_return_blocks returns non-empty decoded blocks");
        if (term == nullptr)
          continue;
        returns.insert(term->src_loc());
      }
    }
  }
  return returns;
}

} // namespace rocjitsu
