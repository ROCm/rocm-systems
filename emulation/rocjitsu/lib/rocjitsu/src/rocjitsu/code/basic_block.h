// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file basic_block.h
/// @brief Basic block representation for decoded instruction sequences.

#ifndef ROCJITSU_CODE_BASIC_BLOCK_H_
#define ROCJITSU_CODE_BASIC_BLOCK_H_

#include "rocjitsu/code/instruction_list.h"
#include "util/intrusive_list.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

class CodeObject;
class Decoder;

/// @brief A maximal sequence of instructions with single entry, single exit.
///
/// @details Instructions within a basic block execute sequentially. The block ends
/// at a branch, conditional branch, or program terminator instruction.
class BasicBlock : public util::IListNode<BasicBlock> {
public:
  /// @brief Construct a basic block starting at the given byte offset.
  /// @param[in] start_offset Byte offset of the first instruction in the code object.
  explicit BasicBlock(uint64_t start_offset);

  /// @brief Byte offset of the first instruction.
  /// @returns Start offset within the code object.
  uint64_t start_offset() const { return start_offset_; }

  /// @brief Byte offset one past the last instruction.
  /// @returns End offset (start_offset + size).
  uint64_t end_offset() const { return start_offset_ + size_; }

  /// @brief Total size of the basic block in bytes.
  /// @returns Size in bytes.
  uint32_t size() const { return size_; }

  /// @brief Number of decoded instructions in the block.
  /// @returns Instruction count.
  uint32_t num_instructions() const { return num_instructions_; }

  /// @brief Whether the block ends with a terminator instruction.
  /// @retval true The last instruction is a branch or program terminator.
  /// @retval false The block falls through to the next.
  bool has_terminator() const { return has_terminator_; }

  /// @brief Last instruction in the block, or nullptr for an empty block.
  [[nodiscard]] const Instruction *terminator() const;

  /// @brief CFG successor blocks.
  ///
  /// @details Edges are non-owning links between blocks returned by build().
  /// They are valid as long as the returned vector owns the blocks.
  [[nodiscard]] const std::vector<BasicBlock *> &successors() const { return successors_; }

  /// @brief CFG predecessor blocks, inverse of successors().
  [[nodiscard]] const std::vector<BasicBlock *> &predecessors() const { return predecessors_; }

  /// @brief Mutable access to the intrusive list of instructions.
  /// @returns Reference to the instruction list.
  InstructionList &instructions() { return instructions_; }

  /// @brief Const access to the intrusive list of instructions.
  /// @returns Const reference to the instruction list.
  const InstructionList &instructions() const { return instructions_; }

  /// @brief Build basic blocks from a code object's .text sections.
  /// @param[in] co Code object to analyze.
  /// @param[in] decoder Decoder for the target ISA.
  /// @returns Ordered list of basic blocks with their decoded instructions.
  static std::vector<std::unique_ptr<BasicBlock>> build(const CodeObject &co, Decoder &decoder);

  /// @brief Build basic blocks with additional externally-known entry leaders.
  ///
  /// @param[in] co Code object to analyze.
  /// @param[in] decoder Decoder for the target ISA.
  /// @param[in] extra_leaders Byte offsets that must start a basic block.
  /// @returns Ordered list of basic blocks with their decoded instructions.
  static std::vector<std::unique_ptr<BasicBlock>> build(const CodeObject &co, Decoder &decoder,
                                                        std::span<const uint64_t> extra_leaders);

private:
  void add_instruction(std::unique_ptr<Instruction> inst);
  void add_successor(BasicBlock &successor);

  uint64_t start_offset_;
  uint32_t size_ = 0;
  uint32_t num_instructions_ = 0;
  bool has_terminator_ = false;
  InstructionList instructions_;
  std::vector<std::unique_ptr<Instruction>> storage_;
  std::vector<BasicBlock *> successors_;
  std::vector<BasicBlock *> predecessors_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_BASIC_BLOCK_H_
