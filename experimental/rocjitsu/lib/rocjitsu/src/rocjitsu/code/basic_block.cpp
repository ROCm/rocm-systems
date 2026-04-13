// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"

#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <memory>
#include <string_view>
#include <vector>

namespace rocjitsu {

namespace {

/// @brief Determine if an instruction terminates a basic block.
///
/// Checks both the InstFlags bitmask (BRANCH, COND_BRANCH, PROGRAM_TERMINATOR)
/// and the mnemonic as a fallback, since not all ISA backends set flags yet.
bool is_block_terminator(const Instruction &inst) {
  if (inst.is_branch())
    return true;

  std::string_view m = inst.mnemonic();
  return m.starts_with("s_endpgm") || m.starts_with("s_branch") || m.starts_with("s_cbranch");
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

std::vector<std::unique_ptr<BasicBlock>> BasicBlock::build(const CodeObject &co, Decoder &decoder) {
  std::vector<std::unique_ptr<BasicBlock>> blocks;

  for (const auto *sec : co.text_sections()) {
    const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
    std::size_t inst_data_size = sec->size() / sizeof(uint32_t);
    uint64_t pc = 0;
    uint64_t byte_offset = 0;

    auto current = std::make_unique<BasicBlock>(byte_offset);

    while (pc < inst_data_size) {
      auto inst = decoder.decode(&inst_data[pc]);
      uint32_t inst_size_bytes = static_cast<uint32_t>(inst->size());
      uint32_t inst_words = inst_size_bytes / sizeof(uint32_t);

      bool terminates = is_block_terminator(*inst);

      current->add_instruction(std::move(inst));
      pc += inst_words;
      byte_offset += inst_size_bytes;

      if (terminates) {
        blocks.push_back(std::move(current));
        if (pc < inst_data_size)
          current = std::make_unique<BasicBlock>(byte_offset);
      }
    }

    if (current && current->num_instructions() > 0)
      blocks.push_back(std::move(current));
  }

  return blocks;
}

} // namespace rocjitsu
