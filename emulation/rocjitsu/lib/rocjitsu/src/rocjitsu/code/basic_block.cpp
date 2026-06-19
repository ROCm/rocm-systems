// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"

#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

namespace {

bool is_block_terminator(const Instruction &inst) {
  return inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR);
}

bool has_no_static_successor(const Instruction &inst) {
  // Indirect calls return to the fallthrough block; indirect branches do not
  // expose a statically-known successor in this local CFG.
  return inst.flags() & (PROGRAM_TERMINATOR | INDIRECT_BRANCH);
}

bool is_unconditional_branch(const Instruction &inst) {
  return (inst.flags() & BRANCH) && !(inst.flags() & COND_BRANCH);
}

} // namespace

BasicBlock::BasicBlock(uint64_t start_offset) : start_offset_(start_offset) {}

void BasicBlock::add_instruction(std::unique_ptr<Instruction> inst) {
  size_ += static_cast<uint32_t>(inst->size());
  has_terminator_ = is_block_terminator(*inst);
  ++num_instructions_;
  inst->parent_ = this;
  instructions_.push_back(*inst);
  storage_.push_back(std::move(inst));
}

const Instruction *BasicBlock::terminator() const {
  if (storage_.empty())
    return nullptr;
  return storage_.back().get();
}

void BasicBlock::add_successor(BasicBlock &successor) {
  if (std::ranges::find(successors_, &successor) != successors_.end())
    return;
  successors_.push_back(&successor);
  successor.predecessors_.push_back(this);
}

std::vector<std::unique_ptr<BasicBlock>> BasicBlock::build(const CodeObject &co, Decoder &decoder) {
  return build(co, decoder, {});
}

std::vector<std::unique_ptr<BasicBlock>>
BasicBlock::build(const CodeObject &co, Decoder &decoder, std::span<const uint64_t> extra_leaders) {
  std::vector<std::unique_ptr<BasicBlock>> blocks;

  for (const auto *sec : co.text_sections()) {
    const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
    std::size_t inst_data_size = sec->size() / sizeof(uint32_t);
    uint64_t pc = 0;
    uint64_t byte_offset = 0;

    struct DecodedInst {
      uint64_t offset;
      std::unique_ptr<Instruction> inst;
    };
    std::vector<DecodedInst> decoded;

    while (pc < inst_data_size) {
      auto *raw_inst = decoder.decode(&inst_data[pc]);
      std::unique_ptr<Instruction> inst(raw_inst);
      uint32_t inst_size_bytes = static_cast<uint32_t>(inst->size());
      uint32_t inst_words = inst_size_bytes / sizeof(uint32_t);

      decoded.push_back({byte_offset, std::move(inst)});
      pc += inst_words;
      byte_offset += inst_size_bytes;
    }

    if (decoded.empty())
      continue;

    const uint64_t section_end = byte_offset;
    std::set<uint64_t> leaders;
    leaders.insert(decoded.front().offset);
    for (uint64_t leader : extra_leaders) {
      if (leader < section_end)
        leaders.insert(leader);
    }

    // A block has one entry. In addition to splitting after terminators, split
    // at every direct branch target so backwards loop edges and if/else joins
    // point to real BasicBlock objects instead of the middle of a larger block.
    for (size_t i = 0; i < decoded.size(); ++i) {
      const auto &entry = decoded[i];
      const auto &inst = *entry.inst;
      const uint64_t next_offset = entry.offset + static_cast<uint64_t>(inst.size());

      if (is_block_terminator(inst) && next_offset < section_end)
        leaders.insert(next_offset);

      auto branch_delta = inst.branch_offset_bytes();
      assert((!(inst.flags() & (BRANCH | COND_BRANCH)) || branch_delta.has_value()) &&
             "direct branch is missing branch_offset_bytes()");

      if (branch_delta) {
        // AMDGPU direct branch immediates are PC-relative to the next
        // instruction. The generator exposes that delta in bytes.
        const int64_t target =
            static_cast<int64_t>(next_offset) + static_cast<int64_t>(*branch_delta);
        if (target >= 0 && static_cast<uint64_t>(target) < section_end)
          leaders.insert(static_cast<uint64_t>(target));
      }
    }

    std::vector<std::unique_ptr<BasicBlock>> section_blocks;
    for (size_t i = 0; i < decoded.size();) {
      auto current = std::make_unique<BasicBlock>(decoded[i].offset);
      while (i < decoded.size()) {
        const uint64_t inst_offset = decoded[i].offset;
        const uint64_t next_offset = inst_offset + static_cast<uint64_t>(decoded[i].inst->size());
        const bool terminates = is_block_terminator(*decoded[i].inst);
        current->add_instruction(std::move(decoded[i].inst));
        ++i;

        if (terminates || (i < decoded.size() && leaders.contains(next_offset)))
          break;
      }
      section_blocks.push_back(std::move(current));
    }

    std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
    block_by_offset.reserve(section_blocks.size());
    for (auto &block : section_blocks)
      block_by_offset.emplace(block->start_offset(), block.get());

    for (size_t i = 0; i < section_blocks.size(); ++i) {
      auto &block = *section_blocks[i];
      const Instruction *term = block.terminator();
      if (term == nullptr || has_no_static_successor(*term))
        continue;

      auto branch_delta = term->branch_offset_bytes();
      assert((!(term->flags() & (BRANCH | COND_BRANCH)) || branch_delta.has_value()) &&
             "direct branch is missing branch_offset_bytes()");

      if (branch_delta) {
        // BasicBlock::end_offset() is the next instruction address for the
        // terminator, which is the base used by AMDGPU direct branch labels.
        const int64_t target =
            static_cast<int64_t>(block.end_offset()) + static_cast<int64_t>(*branch_delta);
        if (target >= 0) {
          auto target_it = block_by_offset.find(static_cast<uint64_t>(target));
          if (target_it != block_by_offset.end())
            block.add_successor(*target_it->second);
        }
      }

      const auto fallthrough_it = block_by_offset.find(block.end_offset());
      // Conditional branches and ordinary instructions may fall through; direct
      // unconditional branches do not.
      if (!is_unconditional_branch(*term) && fallthrough_it != block_by_offset.end())
        block.add_successor(*fallthrough_it->second);
    }

    for (auto &block : section_blocks)
      blocks.push_back(std::move(block));
  }

  return blocks;
}

} // namespace rocjitsu
