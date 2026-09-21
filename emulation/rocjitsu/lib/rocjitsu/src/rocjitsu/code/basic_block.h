// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file basic_block.h
/// @brief Basic block representation for decoded instruction sequences.

#ifndef ROCJITSU_CODE_BASIC_BLOCK_H_
#define ROCJITSU_CODE_BASIC_BLOCK_H_

#include "rocjitsu/code/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/instruction_list.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decode_result.h"
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
  /// @brief Half-open byte interval within the single text section being analyzed.
  struct CodeRange {
    uint64_t start_offset = 0;
    uint64_t size = 0;
  };

  /// @brief Which instruction stream indirect discovery and finalization consume.
  enum class DecodePolicy : uint8_t {
    /// Decode all text before indirect discovery, preserving eager DBT behavior.
    FullSection,
    /// Decode entries and discovered targets, skipping unrelated bytes.
    Reachable,
  };

  /// @brief Construction inputs with distinct entry states, decode roots and boundaries.
  ///
  /// @details Both policies share indirect discovery and block/call finalization.
  /// Input spans are borrowed only during construction, not retained by the CFG.
  struct BuildOptions {
    /// @brief Select reachable decoding by default; DBT can explicitly retain eager decoding.
    ///
    /// @details FullSection avoids repeated recovery on workloads that inspect all text anyway.
    DecodePolicy decode_policy = DecodePolicy::Reachable;

    /// @brief Section-relative external entry offsets, matching build_reachable's entry_offsets.
    ///
    /// @details These establish external states, unlike seeds and split points. They correspond
    /// to build's extra_leaders. Reachable requires valid, permitted instruction boundaries.
    std::span<const uint64_t> entries = {};

    /// @brief Allowed byte intervals for Reachable; empty permits the entire text section.
    ///
    /// @details FullSection rejects nonempty ranges. See build_reachable for range validation.
    std::span<const CodeRange> permitted_ranges = {};

    /// @brief Optional Reachable code roots, processed after the real entry closure stabilizes.
    ///
    /// @details These also split blocks without creating external states. FullSection rejects
    /// nonempty seeds. They root indirect discovery across block boundaries without weakening
    /// incoming register facts or assuming default mode values. Processed in address order;
    /// each seed's indirect closure completes before considering the next. Seeds inside
    /// established instructions or padding are ignored; other incompatible boundaries and
    /// malformed code fail.
    std::span<const uint64_t> decode_seeds = {};

    /// @brief Block boundaries that neither seed decoding nor introduce external entry states.
    ///
    /// @details Accepted by both policies; these match the wrappers' extra_split_points.
    std::span<const uint64_t> split_points = {};

    /// @brief How discovery identifies external states; Reachable requires ExplicitOnly.
    ///
    /// @details FullSection additionally preserves its historical implicit first external entry.
    /// Unlike these options, the legacy build() wrapper defaults to InferPredecessorless.
    ExternalEntryPolicy entry_policy = ExternalEntryPolicy::ExplicitOnly;
  };

  /// @brief First missing successor, or control flow requiring indirect analysis.
  enum class SuccessorIssue : uint8_t {
    None,
    MissingBranchTarget,
    MissingCallTarget,
    MissingFallthrough,
    MissingCallContinuation,
    IndirectControlFlow,
  };

  /// @brief Kind of call-like edge recorded outside the local CFG successor list.
  enum class CallEdgeKind {
    DirectCall,
    IndirectSwapPc,
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

  /// @brief Whether the block ends with an explicit or implicit terminator.
  /// @retval true The last instruction is a branch/program terminator, or the block carries an
  /// inferred boundary (see has_implicit_terminator()).
  /// @retval false The block falls through to the next.
  bool has_terminator() const { return has_terminator_; }

  /// @brief Whether an inferred boundary terminates this block's fallthrough edge.
  ///
  /// @details Clang may omit an architectural terminator after __builtin_unreachable(). Two
  /// source conditions establish the boundary, and they are equivalent because neither leaves a
  /// next instruction to reach: the following word is gfx1250 zero-filled text padding, or the
  /// block ends at the end of `.text`. Relocation materializes either as an s_endpgm.
  ///
  /// This cuts the FALLTHROUGH edge only. A conditional or indirect branch carrying this flag
  /// still has a live taken edge, so consumers must not read it as a whole-block program exit --
  /// classify_function() treats it as one only when no branch edge remains, which keeps the taken
  /// target subject to the missing-target checks.
  bool has_implicit_terminator() const { return has_implicit_terminator_; }

  /// @brief Last instruction in the block, or nullptr for an empty block.
  [[nodiscard]] const Instruction *terminator() const;

  /// @brief CFG successor blocks.
  ///
  /// @details Edges are local, context-free CFG links between blocks returned
  /// by CFG construction. Function-call targets are exposed through call_edges()
  /// instead of this list because their return flow depends on the call site.
  [[nodiscard]] const std::vector<BasicBlock *> &successors() const { return successors_; }

  /// @brief CFG predecessor blocks, inverse of successors().
  [[nodiscard]] const std::vector<BasicBlock *> &predecessors() const { return predecessors_; }

  /// @brief Whether construction omitted a required edge or deferred indirect flow.
  ///
  /// @details A missing edge does not make decoding fail. Consumers must check this
  /// status before treating the graph as complete. Indirect control flow is marked
  /// conservatively even when some target fixups or call edges are available.
  /// Only the first issue is retained; this is not an exhaustive diagnosis.
  [[nodiscard]] SuccessorIssue successor_issue() const { return successor_issue_; }

  /// @brief Function-call edges that leave this block.
  [[nodiscard]] const std::vector<CallEdge> &call_edges() const { return call_edges_; }

  /// @brief Add a call edge proven by an external finite-target analysis.
  ///
  /// @details Relocation-backed device function tables are discovered after
  /// ordinary block construction. Their dynamic dispatch remains indirect in
  /// the instruction stream, but each populated table slot is a concrete
  /// callee for reachability and kernel-scoped liveness.
  void add_call_edge(CallEdge edge);

  /// @brief Static indirect branch fixup metadata rooted in this block.
  ///
  /// @details Block construction computes these while all decoded instructions
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

  /// @brief Every PC-relative address producer whose `s_getpc_b64` lives in this block.
  ///
  /// @details Unlike static_indirect_call_fixups(), this covers producers that
  /// no recovered consumer references, including ones the pass could not follow.
  /// DBT needs the complete set to decide whether a kernel scope can be made
  /// free of stale PC-derived values.
  [[nodiscard]] const std::vector<PcAddressBuilder> &static_pc_address_builders() const {
    return static_pc_address_builders_;
  }

  /// @brief Mutable access to the intrusive list of instructions.
  /// @returns Reference to the instruction list.
  InstructionList &instructions() { return instructions_; }

  /// @brief Const access to the intrusive list of instructions.
  /// @returns Const reference to the instruction list.
  const InstructionList &instructions() const { return instructions_; }

  /// @brief Build a CFG using the selected decode policy and entry model.
  ///
  /// @details Reachable construction requires ExplicitOnly entry states; FullSection rejects
  /// ranges and decode seeds. See build_reachable() for reachability and completeness contracts.
  /// @param[in] co Code object to analyze; must outlive the returned blocks.
  /// @param[in] decoder Decoder for the selected architecture.
  /// @param[in] arch Architecture used by decoding, discovery and padding policies.
  /// @param[in] options Borrowed construction inputs, used only for the duration of the call.
  /// @param[in] emit_error Destination for validation and offset-bearing decode diagnostics.
  /// @returns Ordered blocks, or failure on invalid inputs or exhausted discovery budgets.
  static FailureOr<std::vector<std::unique_ptr<BasicBlock>>>
  build_cfg(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
            const BuildOptions &options, DecodeErrorEmitter emit_error = {});

  /// @brief Build basic blocks from a code object's .text sections.
  ///
  /// @details Recovered indirect branch targets are added as block leaders before
  /// the block objects are finalized, so users never see a recovered edge whose
  /// destination points into the middle of a larger block. Syntactic call
  /// fallthroughs remain provisional until call-return classification finishes.
  /// @param[in] co Code object to analyze.
  /// @param[in] decoder Decoder for the target ISA.
  /// @param[in] arch ISA architecture used to match static PC builders.
  /// @param[in] emit_error Destination for a decode diagnostic, including the
  /// source `.text` byte offset.
  /// @param[in] extra_leaders Byte offsets that must start a basic block AND are entered from
  /// outside the decoded graph. Under ExplicitOnly these become the external-entry set, so an
  /// offset listed here has no incoming caller facts.
  /// @param[in] entry_policy Whether predecessorless blocks are inferred to be
  /// external function entries. Use ExplicitOnly only when extra_leaders
  /// enumerates every externally reachable entry.
  /// @param[in] extra_split_points Byte offsets that must start a basic block but are NOT external
  /// entries. Function-entry symbols and stored-pointer targets belong here: they are genuine
  /// boundaries, yet most are ordinary helpers whose callers reach them by a decoded edge, and
  /// calling them external would throw those caller facts away.
  /// @returns Ordered basic blocks with their decoded instructions, or failure.
  static FailureOr<std::vector<std::unique_ptr<BasicBlock>>>
  build(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
        DecodeErrorEmitter emit_error = {}, std::span<const uint64_t> extra_leaders = {},
        ExternalEntryPolicy entry_policy = ExternalEntryPolicy::InferPredecessorless,
        std::span<const uint64_t> extra_split_points = {});

  /// @brief Decode reachable text, extending direct control flow with discovered indirect targets.
  ///
  /// @details Requires exactly one text section when entries or decode seeds are nonempty.
  /// No entries or seeds produce an empty graph. Entries and optional permitted ranges are byte
  /// offsets within that section; every entry must lie in a permitted range.
  /// Ranges must be aligned, nonempty and contained in the section; overlapping
  /// or adjacent ranges are united. An empty range list permits the whole section.
  /// Decode fallthrough, direct branches/calls and possible continuations, then
  /// repeat indirect discovery and decoding until no new targets are found.
  /// Unrelated bytes are not decoded. Indirect control flow remains explicit in
  /// successor_issue(): discovering concrete edges alone does not certify a
  /// complete target set or context-specific returns. Edges leaving permitted ranges
  /// also set successor_issue(). Reached malformed/truncated instructions, invalid
  /// entries/ranges and overlapping instruction boundaries return failure and emit
  /// a diagnostic. Exceeding a root closure's bounded discovery rounds also returns failure.
  /// @param[in] co Code object with one text section; must outlive the returned blocks.
  /// @param[in] decoder Decoder for the selected ISA architecture.
  /// @param[in] arch Architecture used by control-flow and padding policies.
  /// @param[in] entry_offsets External entries in section-relative bytes.
  /// @param[in] emit_error Destination for validation and offset-bearing decode diagnostics.
  /// @param[in] permitted_ranges Allowed code intervals; empty permits the whole section.
  /// @param[in] decode_seeds Additional code roots to decode after the entry closure,
  /// without introducing external entry states. These also split blocks. Seeds inside decoded
  /// instructions or padding are ignored; consumers must validate any seed they later use as an
  /// actual entry.
  /// @param[in] extra_split_points Block boundaries that do not seed decoding or external states.
  /// @returns Blocks ordered by text offset, or failure with a diagnostic.
  static FailureOr<std::vector<std::unique_ptr<BasicBlock>>>
  build_reachable(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                  std::span<const uint64_t> entry_offsets, DecodeErrorEmitter emit_error = {},
                  std::span<const CodeRange> permitted_ranges = {},
                  std::span<const uint64_t> decode_seeds = {},
                  std::span<const uint64_t> extra_split_points = {});

private:
  struct DecodedSection;
  /// Consume prepared instructions and discovery facts while retaining permitted_ranges
  /// for padding and missing-edge decisions.
  static FailureOr<std::vector<std::unique_ptr<BasicBlock>>>
  build_impl(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
             DecodeErrorEmitter emit_error, std::span<const uint64_t> extra_leaders,
             ExternalEntryPolicy entry_policy, std::span<const uint64_t> extra_split_points,
             DecodedSection *prepared, std::span<const CodeRange> permitted_ranges);

  void note_successor_issue(SuccessorIssue issue) {
    if (successor_issue_ == SuccessorIssue::None)
      successor_issue_ = issue;
  }

  void add_instruction(std::unique_ptr<Instruction> inst);
  void add_successor(BasicBlock &successor);
  /// Remove one proven-dead edge while preserving the inverse predecessor list.
  [[nodiscard]] bool remove_successor(BasicBlock &successor);
  void add_static_indirect_call_fixup(IndirectCallFixup fixup);
  void add_static_pc_address_builder(PcAddressBuilder builder);

  SuccessorIssue successor_issue_ = SuccessorIssue::None;
  uint64_t start_offset_;
  uint32_t size_ = 0;
  uint32_t num_instructions_ = 0;
  bool has_terminator_ = false;
  bool has_implicit_terminator_ = false;
  InstructionList instructions_;
  std::vector<std::unique_ptr<Instruction>> storage_;
  std::vector<BasicBlock *> successors_;
  std::vector<BasicBlock *> predecessors_;
  std::vector<CallEdge> call_edges_;
  std::vector<IndirectCallFixup> static_indirect_call_fixups_;
  std::vector<PcAddressBuilder> static_pc_address_builders_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_BASIC_BLOCK_H_
