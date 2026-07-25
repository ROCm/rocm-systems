// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file basic_block.h
/// @brief Basic block representation for decoded instruction sequences.

#ifndef ROCJITSU_CODE_BASIC_BLOCK_H_
#define ROCJITSU_CODE_BASIC_BLOCK_H_

#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/instruction_list.h"
#include "rocjitsu/code/rj_code.h"
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
  /// @brief Half-open byte range in `.text` known to contain instructions.
  struct CodeRange {
    uint64_t start_offset = 0;
    uint64_t size = 0;
  };

  /// @brief Kind of call-like edge recorded outside the local CFG successor list.
  enum class CallEdgeKind {
    DirectCall,
    IndirectSwapPc,
  };

  /// @brief First structurally-known CFG edge omitted from the decoded graph.
  enum class StaticSuccessorIssue : uint8_t {
    None,
    MissingBranchTarget,
    MissingCallTarget,
    MissingFallthrough,
    MissingCallContinuation,
  };

  /// @brief Context-sensitive call edge from this block to a function entry.
  ///
  /// @details Calls are intentionally not ordinary BasicBlock successors. The
  /// callee body can be shared by multiple kernels or call sites, while the
  /// return continuation is call-context-specific. DBT builds a kernel-local
  /// graph from these records when it needs interprocedural reachability or
  /// liveness, adding return edges only for call sites in the current scope.
  struct CallEdge {
    CallEdgeKind kind = CallEdgeKind::IndirectSwapPc;
    BasicBlock *callee = nullptr;
    BasicBlock *continuation = nullptr;
    uint64_t source_call_offset = 0;
    uint16_t return_sreg = 0;
  };

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
  /// @details Edges are local, context-free CFG links between blocks returned
  /// by build(). Function-call targets are exposed through call_edges() instead
  /// of this list because their return flow depends on the call site.
  [[nodiscard]] const std::vector<BasicBlock *> &successors() const { return successors_; }

  /// @brief CFG predecessor blocks, inverse of successors().
  [[nodiscard]] const std::vector<BasicBlock *> &predecessors() const { return predecessors_; }

  /// @brief Whether every required direct target and fallthrough was decoded.
  ///
  /// @details This covers only structurally known edges. Indirect target
  /// exhaustiveness and context-sensitive returns remain client-specific
  /// checks. A false result means the decoded graph omitted an edge that the
  /// instruction stream can take, for example at a symbol-range boundary.
  [[nodiscard]] bool static_successors_complete() const {
    return static_successor_issue_ == StaticSuccessorIssue::None;
  }

  /// @brief Why a required structurally-known successor was omitted.
  ///
  /// @details The first issue is retained so clients can distinguish missing
  /// branch targets, call targets, ordinary fallthroughs, and call
  /// continuations without re-decoding the terminator.
  [[nodiscard]] StaticSuccessorIssue static_successor_issue() const {
    return static_successor_issue_;
  }

  /// @brief Function-call edges that leave this block.
  [[nodiscard]] const std::vector<CallEdge> &call_edges() const { return call_edges_; }

  /// @brief Static indirect branch fixup metadata rooted in this block.
  ///
  /// @details BasicBlock::build() computes these while all decoded instructions
  /// and source offsets are still adjacent. The fixups are grouped on the block
  /// that contains the recovered setpc/swappc consumer. The same target metadata
  /// may become either an ordinary CFG successor or a call_edges() record:
  /// ordinary setpc-style branches are context-free successors, while validated
  /// swappc calls are kept out of successors() so each kernel scope can add only
  /// the return continuation that belongs to its call site. DBT also uses the
  /// fixup bytes to rewrite the original address-builder words after relocation.
  [[nodiscard]] const std::vector<IndirectCallFixup> &static_indirect_call_fixups() const {
    return static_indirect_call_fixups_;
  }

  /// @brief Mutable access to the intrusive list of instructions.
  /// @returns Reference to the instruction list.
  InstructionList &instructions() { return instructions_; }

  /// @brief Const access to the intrusive list of instructions.
  /// @returns Const reference to the instruction list.
  const InstructionList &instructions() const { return instructions_; }

  /// @brief Build basic blocks from a code object's .text sections.
  ///
  /// @details Recovered indirect branch targets are added as block leaders before
  /// the block objects are finalized, so users never see a recovered edge whose
  /// destination points into the middle of a larger block.
  /// @param[in] co Code object to analyze.
  /// @param[in] decoder Decoder for the target ISA.
  /// @param[in] arch ISA architecture used to match static PC builders.
  /// @param[in] extra_leaders Byte offsets that must start a basic block.
  /// @param[in] code_ranges Optional symbol-backed instruction ranges. When
  /// empty, decode the complete text section. Supplying ranges excludes
  /// alignment padding and embedded data from instruction decoding.
  /// @returns Ordered list of basic blocks with their decoded instructions.
  static std::vector<std::unique_ptr<BasicBlock>>
  build(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
        std::span<const uint64_t> extra_leaders = {}, std::span<const CodeRange> code_ranges = {});

  /// @brief Build only basic blocks reachable from externally-known entries.
  ///
  /// @details Unlike build(), this avoids decoding padding and data embedded in
  /// executable sections of library code objects. Statically recoverable call
  /// targets are decoded on demand so callers can model helper waits without
  /// scanning the rest of a large library code object.
  /// @param[in] wavefront_size Kernel wave size when known, or zero to retain
  /// decoded maximum-width wave-mask operands during indirect-edge discovery.
  static std::vector<std::unique_ptr<BasicBlock>>
  build_reachable(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                  std::span<const uint64_t> entry_offsets,
                  std::span<const uint64_t> entry_sizes = {}, uint32_t wavefront_size = 0);

private:
  void add_instruction(std::unique_ptr<Instruction> inst);
  void add_successor(BasicBlock &successor);
  void add_call_edge(CallEdge edge);
  void add_static_indirect_call_fixup(IndirectCallFixup fixup);
  void note_static_successor_issue(StaticSuccessorIssue issue) {
    if (static_successor_issue_ == StaticSuccessorIssue::None)
      static_successor_issue_ = issue;
  }

  uint64_t start_offset_;
  uint32_t size_ = 0;
  uint32_t num_instructions_ = 0;
  bool has_terminator_ = false;
  InstructionList instructions_;
  std::vector<std::unique_ptr<Instruction>> storage_;
  std::vector<BasicBlock *> successors_;
  std::vector<BasicBlock *> predecessors_;
  std::vector<CallEdge> call_edges_;
  std::vector<IndirectCallFixup> static_indirect_call_fixups_;
  StaticSuccessorIssue static_successor_issue_ = StaticSuccessorIssue::None;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_BASIC_BLOCK_H_
