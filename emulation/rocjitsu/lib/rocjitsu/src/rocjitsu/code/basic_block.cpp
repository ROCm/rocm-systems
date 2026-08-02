// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"

#include "rocjitsu/analysis/control_flow.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include "util/except.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

namespace {

bool is_block_terminator(const Instruction &inst) {
  return is_program_path_terminator(inst) ||
         (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL));
}

bool has_no_static_successor(const Instruction &inst) {
  // Indirect calls return to the fallthrough block; indirect branches do not
  // expose a statically-known successor in this local CFG.
  return is_program_path_terminator(inst) || (inst.flags() & INDIRECT_BRANCH);
}

bool is_unconditional_branch(const Instruction &inst) {
  return (inst.flags() & BRANCH) && !(inst.flags() & COND_BRANCH);
}

uint32_t first_word(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  return raw == nullptr ? 0 : raw[0];
}

bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  // gfx1250 renamed the scalar PC transfer without changing its role in the
  // call/return CFG. Accept both spellings; the raw source operand remains in
  // the low byte for the canonical one-word form.
  const std::string_view mnemonic = inst.mnemonic();
  if (inst.size() != sizeof(uint32_t) || (mnemonic != "s_setpc_b64" && mnemonic != "s_set_pc_i64"))
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

bool has_exact_words(const Instruction &inst, std::span<const uint32_t> words) {
  if (inst.size() != static_cast<int>(words.size_bytes()) || inst.raw_encoding() == nullptr)
    return false;
  return std::equal(words.begin(), words.end(), inst.raw_encoding());
}

std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

enum class CallReturnClassification {
  Unknown,
  Returning,
  NonReturning,
};

struct DeferredCallTarget {
  BasicBlock::CallEdgeKind kind = BasicBlock::CallEdgeKind::IndirectSwapPc;
  BasicBlock *target = nullptr;
  uint64_t source_call_offset = 0;
};

struct DeferredCallSite {
  BasicBlock *source = nullptr;
  BasicBlock *continuation = nullptr;
  uint16_t return_sreg = 0;
  bool target_set_incomplete = false;
  std::vector<DeferredCallTarget> targets;
};

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

bool BasicBlock::is_gfx1250_clang_unreachable_stub() const {
  // clang emits two known rocPRIM target-specialization stubs for a source body
  // ending in __builtin_unreachable(). Neither has an architectural terminator;
  // its trailing zero is function-alignment padding. Match the complete decoded
  // body so an arbitrary reachable instruction followed by zero remains invalid.
  constexpr std::array<uint32_t, 2> kSetReplayMode = {0xb9800641u, 1u};
  if (storage_.size() == 1)
    return has_exact_words(*storage_[0], kSetReplayMode);

  if (storage_.size() != 3)
    return false;
  constexpr std::array<uint32_t, 3> kGlobalPrefetchB8 = {0xee174000u, 0x00040000u, 0u};
  constexpr std::array<uint32_t, 1> kVNop = {0x7e000000u};
  return has_exact_words(*storage_[0], kGlobalPrefetchB8) && has_exact_words(*storage_[1], kVNop) &&
         has_exact_words(*storage_[2], kSetReplayMode);
}

void BasicBlock::add_successor(BasicBlock &successor) {
  if (std::ranges::find(successors_, &successor) != successors_.end())
    return;
  successors_.push_back(&successor);
  successor.predecessors_.push_back(this);
}

bool BasicBlock::remove_successor(BasicBlock &successor) {
  const auto successor_it = std::ranges::find(successors_, &successor);
  if (successor_it == successors_.end())
    return false;
  successors_.erase(successor_it);

  const auto predecessor_it = std::ranges::find(successor.predecessors_, this);
  assert(predecessor_it != successor.predecessors_.end() &&
         "successor/predecessor edges must remain inverse");
  successor.predecessors_.erase(predecessor_it);
  return true;
}

void BasicBlock::add_call_edge(CallEdge edge) {
  if (edge.callee == nullptr || edge.continuation == nullptr)
    return;
  const auto duplicate = std::ranges::find_if(call_edges_, [&](const CallEdge &existing) {
    return existing.kind == edge.kind && existing.callee == edge.callee &&
           existing.continuation == edge.continuation &&
           existing.source_call_offset == edge.source_call_offset &&
           existing.return_sreg == edge.return_sreg;
  });
  if (duplicate != call_edges_.end())
    return;
  call_edges_.push_back(edge);
}

void BasicBlock::add_static_indirect_call_fixup(IndirectCallFixup fixup) {
  static_indirect_call_fixups_.push_back(fixup);
}

void BasicBlock::add_static_pc_address_builder(PcAddressBuilder builder) {
  static_pc_address_builders_.push_back(builder);
}

std::vector<std::unique_ptr<BasicBlock>> BasicBlock::build(const CodeObject &co, Decoder &decoder,
                                                           rj_code_arch_t arch,
                                                           std::span<const uint64_t> extra_leaders,
                                                           ExternalEntryPolicy entry_policy,
                                                           std::span<const CodeRange> code_ranges) {
  return build_impl(co, decoder, arch, extra_leaders, entry_policy, code_ranges, {});
}

std::vector<std::unique_ptr<BasicBlock>>
BasicBlock::build_impl(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                       std::span<const uint64_t> extra_leaders, ExternalEntryPolicy entry_policy,
                       std::span<const CodeRange> code_ranges,
                       std::span<const IndirectCallFixup> retained_indirect_targets) {
  std::vector<std::unique_ptr<BasicBlock>> blocks;

  for (const auto *sec : co.text_sections()) {
    const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
    std::vector<std::unique_ptr<Instruction>> decoded;

    std::vector<CodeRange> section_ranges;
    if (code_ranges.empty()) {
      section_ranges.push_back({.start_offset = 0, .size = sec->size()});
    } else {
      section_ranges.reserve(code_ranges.size());
      for (const CodeRange &range : code_ranges) {
        if (range.size == 0 || range.start_offset >= sec->size())
          continue;
        if (range.start_offset % sizeof(uint32_t) != 0 || range.size % sizeof(uint32_t) != 0)
          throw util::InvalidInst("unaligned code range", "Invalid CFG: ");
        const uint64_t available = sec->size() - range.start_offset;
        section_ranges.push_back(
            {.start_offset = range.start_offset, .size = std::min(range.size, available)});
      }
      std::ranges::sort(section_ranges, {}, &CodeRange::start_offset);
      std::vector<CodeRange> merged_ranges;
      merged_ranges.reserve(section_ranges.size());
      for (const CodeRange &range : section_ranges) {
        if (merged_ranges.empty() ||
            range.start_offset > merged_ranges.back().start_offset + merged_ranges.back().size) {
          merged_ranges.push_back(range);
          continue;
        }
        CodeRange &previous = merged_ranges.back();
        const uint64_t merged_end =
            std::max(previous.start_offset + previous.size, range.start_offset + range.size);
        previous.size = merged_end - previous.start_offset;
      }
      section_ranges = std::move(merged_ranges);
    }

    for (const CodeRange &range : section_ranges) {
      uint64_t byte_offset = range.start_offset;
      const uint64_t range_end = range.start_offset + range.size;
      while (byte_offset < range_end) {
        const size_t pc = static_cast<size_t>(byte_offset / sizeof(uint32_t));
        // gfx1250 code objects place zero-filled alignment holes between function
        // bodies. Zero is not an instruction, so skip it before decoding.
        if (arch == ROCJITSU_CODE_ARCH_GFX1250 && inst_data[pc] == 0) {
          byte_offset += sizeof(uint32_t);
          continue;
        }

        Instruction *raw_inst = nullptr;
        try {
          const std::size_t remaining_words =
              static_cast<std::size_t>((range_end - byte_offset) / sizeof(uint32_t));
          raw_inst = decoder.decode_window(
              std::span<const uint32_t>(&inst_data[pc], remaining_words), byte_offset);
        } catch (const util::InvalidInst &error) {
          throw util::InvalidInst(std::string(error.what()) + " at .text byte offset " +
                                      std::to_string(byte_offset),
                                  "");
        }
        std::unique_ptr<Instruction> inst(raw_inst);
        const uint32_t inst_size_bytes = static_cast<uint32_t>(inst->size());
        if (inst_size_bytes == 0 || byte_offset + inst_size_bytes > range_end)
          throw util::InvalidInst("truncated instruction", "Invalid CFG: ");

        decoded.push_back(std::move(inst));
        byte_offset += inst_size_bytes;
      }
    }

    if (decoded.empty())
      continue;

    std::vector<const Instruction *> decoded_insts;
    decoded_insts.reserve(decoded.size());
    for (const auto &inst : decoded)
      decoded_insts.push_back(inst.get());

    const uint64_t section_end = sec->size();
    // Indirect target discovery belongs with block construction because
    // recovered branch targets must become leaders before instructions are
    // moved into final BasicBlock storage. The discovery pass first walks the
    // direct CFG and only records an indirect edge when the s_getpc-built SGPR
    // pair still has a concrete value at the setpc/swappc consumer.
    const auto text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(sec->data()), sec->size());
    const auto decoded_span =
        std::span<const Instruction *const>(decoded_insts.data(), decoded_insts.size());
    std::vector<PcAddressBuilder> pc_address_builders;
    std::vector<IndirectCallFixup> recovered_indirect_targets = discover_indirect_branch_edges(
        decoded_span, text, arch, extra_leaders, entry_policy, &pc_address_builders);

    const auto same_fixup = [](const IndirectCallFixup &left, const IndirectCallFixup &right) {
      return left.source_call_offset == right.source_call_offset &&
             left.source_target_offset == right.source_target_offset &&
             left.source_call_selector == right.source_call_selector &&
             left.source_call_carrier == right.source_call_carrier &&
             left.source_return_selector == right.source_return_selector;
    };
    for (const IndirectCallFixup &retained : retained_indirect_targets) {
      const auto duplicate =
          std::ranges::find_if(recovered_indirect_targets, [&](const IndirectCallFixup &candidate) {
            return same_fixup(candidate, retained);
          });
      if (duplicate == recovered_indirect_targets.end()) {
        recovered_indirect_targets.push_back(retained);
      } else {
        duplicate->source_incomplete |= retained.source_incomplete;
        duplicate->source_targets_exhaustive &= retained.source_targets_exhaustive;
      }
    }

    std::set<uint64_t> leaders;
    leaders.insert(decoded.front()->src_loc());
    for (uint64_t leader : extra_leaders) {
      if (leader < section_end)
        leaders.insert(leader);
    }
    for (const IndirectCallFixup &fixup : recovered_indirect_targets) {
      if (fixup.source_call_offset < section_end)
        leaders.insert(fixup.source_call_offset);
      if (fixup.source_target_offset < section_end)
        leaders.insert(fixup.source_target_offset);
    }

    // A block has one entry. In addition to splitting after terminators, split
    // at every direct branch target so backwards loop edges and if/else joins
    // point to real BasicBlock objects instead of the middle of a larger block.
    for (size_t i = 0; i < decoded.size(); ++i) {
      const auto &inst = *decoded[i];
      const uint64_t next_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());

      // An undecodable run is a real CFG boundary. Without this leader the
      // block size would collapse the gap and corrupt every following source
      // offset during relocation.
      if (i + 1 < decoded.size() && decoded[i + 1]->src_loc() != next_offset)
        leaders.insert(decoded[i + 1]->src_loc());

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
      auto current = std::make_unique<BasicBlock>(decoded[i]->src_loc());
      while (i < decoded.size()) {
        const uint64_t inst_offset = decoded[i]->src_loc();
        const uint64_t next_offset = inst_offset + static_cast<uint64_t>(decoded[i]->size());
        const bool terminates = is_block_terminator(*decoded[i]);
        current->add_instruction(std::move(decoded[i]));
        ++i;

        const bool decoded_section_end = i >= decoded.size() && next_offset == section_end;
        const bool decode_gap =
            i >= decoded.size() ? next_offset < section_end : decoded[i]->src_loc() != next_offset;
        if (!terminates) {
          const bool reaches_gfx1250_zero = decode_gap && arch == ROCJITSU_CODE_ARCH_GFX1250 &&
                                            next_offset < section_end &&
                                            inst_data[next_offset / sizeof(uint32_t)] == 0;
          const bool is_gfx1250_section_end_stub =
              decoded_section_end && arch == ROCJITSU_CODE_ARCH_GFX1250;
          if ((reaches_gfx1250_zero || is_gfx1250_section_end_stub) &&
              current->is_gfx1250_clang_unreachable_stub()) {
            current->has_terminator_ = true;
            current->has_implicit_terminator_ = true;
          } else if (decode_gap) {
            current->falls_through_to_undecodable_text_ = true;
          }
        }

        if (terminates || decode_gap || (i < decoded.size() && leaders.contains(next_offset)))
          break;
      }
      section_blocks.push_back(std::move(current));
    }

    std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
    block_by_offset.reserve(section_blocks.size());
    for (auto &block : section_blocks)
      block_by_offset.emplace(block->start_offset(), block.get());

    // Attach every discovered PC-relative address producer to the block that
    // contains its s_getpc_b64. Blocks are in ascending source order and cover
    // the decoded stream without overlap, so the owning block is the last one
    // starting at or before the producer.
    const auto starts_after = [](uint64_t offset, const std::unique_ptr<BasicBlock> &block) {
      return offset < block->start_offset();
    };
    for (const PcAddressBuilder &builder : pc_address_builders) {
      const auto it = std::upper_bound(section_blocks.begin(), section_blocks.end(),
                                       builder.source_getpc_offset, starts_after);
      if (it == section_blocks.begin())
        continue;
      BasicBlock &owner = **(it - 1);
      if (builder.source_getpc_offset >= owner.end_offset())
        continue;
      owner.add_static_pc_address_builder(builder);
    }

    std::vector<DeferredCallSite> deferred_calls;
    std::unordered_map<const BasicBlock *, size_t> call_site_by_source;
    auto defer_call = [&](BasicBlock &source, BasicBlock &target, BasicBlock &continuation,
                          CallEdgeKind kind, uint64_t source_call_offset, uint16_t return_sreg,
                          bool target_set_incomplete) {
      const auto [map_it, inserted] =
          call_site_by_source.try_emplace(&source, deferred_calls.size());
      if (inserted) {
        deferred_calls.push_back({.source = &source,
                                  .continuation = &continuation,
                                  .return_sreg = return_sreg,
                                  .target_set_incomplete = target_set_incomplete,
                                  .targets = {}});
      } else {
        DeferredCallSite &site = deferred_calls[map_it->second];
        if (site.continuation != &continuation || site.return_sreg != return_sreg)
          throw util::Exception(
              std::string_view("inconsistent deferred call metadata for one terminator"));
        site.target_set_incomplete |= target_set_incomplete;
      }

      DeferredCallSite &site = deferred_calls[map_it->second];
      const auto duplicate =
          std::ranges::find_if(site.targets, [&](const DeferredCallTarget &existing) {
            return existing.kind == kind && existing.target == &target &&
                   existing.source_call_offset == source_call_offset;
          });
      if (duplicate == site.targets.end())
        site.targets.push_back(
            {.kind = kind, .target = &target, .source_call_offset = source_call_offset});
    };

    for (const IndirectCallFixup &fixup : recovered_indirect_targets) {
      auto source_it = block_by_offset.find(fixup.source_call_offset);
      if (source_it == block_by_offset.end())
        continue;

      BasicBlock *source = source_it->second;
      source->add_static_indirect_call_fixup(fixup);
      if (auto target_it = block_by_offset.find(fixup.source_target_offset);
          target_it != block_by_offset.end()) {
        BasicBlock *target = target_it->second;
        BasicBlock *continuation = nullptr;
        if (auto continuation_it = block_by_offset.find(source->end_offset());
            continuation_it != block_by_offset.end()) {
          continuation = continuation_it->second;
        }

        if (fixup.source_is_call && continuation != nullptr) {
          // Whether this swappc is a function call depends on the callee body
          // reaching a matching setpc return. Defer that question until after
          // ordinary direct CFG edges have been added for every block; otherwise
          // helpers that branch or fall through internally to their return block
          // look falsely non-returning.
          defer_call(*source, *target, *continuation, CallEdgeKind::IndirectSwapPc,
                     fixup.source_call_offset, fixup.source_return_sreg, fixup.source_incomplete);
        } else {
          // Non-call recovered setpc targets are ordinary local CFG edges. If a
          // swappc has no statically-known continuation, keep the old
          // conservative reachability edge instead of pretending it has
          // call/return semantics.
          source->add_successor(*target);
        }
      } else {
        source->note_static_successor_issue(fixup.source_is_call
                                                ? StaticSuccessorIssue::MissingCallTarget
                                                : StaticSuccessorIssue::MissingBranchTarget);
      }
      if (fixup.source_is_call && !block_by_offset.contains(source->end_offset()))
        source->note_static_successor_issue(StaticSuccessorIssue::MissingCallContinuation);
    }

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
          const auto fallthrough_it = block_by_offset.find(block.end_offset());
          const auto call_sdst = s_call_sdst(*term, first_word(*term));
          if (call_sdst && target_it != block_by_offset.end() &&
              fallthrough_it != block_by_offset.end()) {
            // Like recovered swappc, direct s_call validation needs the callee's
            // internal CFG to be complete before we decide whether the target is
            // a returning helper or an ordinary reachable branch target.
            defer_call(block, *target_it->second, *fallthrough_it->second, CallEdgeKind::DirectCall,
                       term->src_loc(), *call_sdst, false);
          } else if (target_it != block_by_offset.end()) {
            block.add_successor(*target_it->second);
          } else {
            block.note_static_successor_issue(call_sdst
                                                  ? StaticSuccessorIssue::MissingCallTarget
                                                  : StaticSuccessorIssue::MissingBranchTarget);
          }
        } else {
          block.note_static_successor_issue(StaticSuccessorIssue::MissingBranchTarget);
        }
      }

      const auto fallthrough_it = block_by_offset.find(block.end_offset());
      // Conditional branches and ordinary instructions may fall through; direct
      // unconditional branches do not.
      if (!is_unconditional_branch(*term) && fallthrough_it != block_by_offset.end())
        block.add_successor(*fallthrough_it->second);
      else if (!is_unconditional_branch(*term) && !is_program_path_terminator(*term) &&
               !block.has_implicit_terminator())
        block.note_static_successor_issue((term->flags() & INDIRECT_CALL) != 0
                                              ? StaticSuccessorIssue::MissingCallContinuation
                                              : StaticSuccessorIssue::MissingFallthrough);
    }

    std::unordered_set<uint64_t> kernel_entry_offsets(extra_leaders.begin(), extra_leaders.end());
    std::vector<std::vector<CallReturnClassification>> classifications;
    classifications.reserve(deferred_calls.size());
    for (const DeferredCallSite &site : deferred_calls)
      classifications.emplace_back(site.targets.size(), CallReturnClassification::Unknown);

    auto classify_function = [&](BasicBlock &callee, uint16_t return_sreg,
                                 const std::vector<std::vector<CallReturnClassification>> &known) {
      struct WalkPoint {
        BasicBlock *block = nullptr;
        std::optional<uint16_t> terminal_return_sreg;
      };
      std::vector<WalkPoint> stack{{.block = &callee, .terminal_return_sreg = std::nullopt}};
      std::set<std::pair<BasicBlock *, std::optional<uint16_t>>> visited;
      bool has_unknown_exit = false;

      auto push_within_function = [&](BasicBlock *successor,
                                      std::optional<uint16_t> terminal_return_sreg) {
        if (successor == nullptr)
          return;
        if (kernel_entry_offsets.contains(successor->start_offset()) && successor != &callee) {
          has_unknown_exit = true;
          return;
        }
        stack.push_back({.block = successor, .terminal_return_sreg = terminal_return_sreg});
      };

      while (!stack.empty()) {
        const WalkPoint point = stack.back();
        stack.pop_back();
        BasicBlock *block = point.block;
        if (block == nullptr || !visited.insert({block, point.terminal_return_sreg}).second)
          continue;

        const Instruction *term = block->terminator();
        if (term == nullptr) {
          has_unknown_exit = true;
          continue;
        }
        if (point.terminal_return_sreg &&
            s_setpc_from_sreg(*term, first_word(*term), *point.terminal_return_sreg)) {
          // This path is the normal return from a nested callee whose body is
          // also being scanned for a direct return through the enclosing pair.
          continue;
        }
        if (s_setpc_from_sreg(*term, first_word(*term), return_sreg))
          return CallReturnClassification::Returning;

        if (auto site_it = call_site_by_source.find(block); site_it != call_site_by_source.end()) {
          const size_t site_index = site_it->second;
          const DeferredCallSite &site = deferred_calls[site_index];
          has_unknown_exit |= site.target_set_incomplete;
          bool reaches_continuation = false;
          for (size_t target_index = 0; target_index < site.targets.size(); ++target_index) {
            switch (known[site_index][target_index]) {
            case CallReturnClassification::Returning:
              reaches_continuation = true;
              push_within_function(site.targets[target_index].target, site.return_sreg);
              break;
            case CallReturnClassification::NonReturning:
              push_within_function(site.targets[target_index].target, point.terminal_return_sreg);
              break;
            case CallReturnClassification::Unknown:
              has_unknown_exit = true;
              break;
            }
          }
          if (reaches_continuation)
            push_within_function(site.continuation, point.terminal_return_sreg);
          for (BasicBlock *successor : block->successors()) {
            if (successor != site.continuation)
              push_within_function(successor, point.terminal_return_sreg);
          }
          continue;
        }

        if (block->falls_through_to_undecodable_text())
          has_unknown_exit = true;

        const bool is_program_exit =
            block->has_implicit_terminator() || is_program_path_terminator(*term);
        const uint64_t flags = term->flags();
        if ((flags & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0) {
          const auto &fixups = block->static_indirect_call_fixups();
          if (fixups.empty() || std::ranges::any_of(fixups, &IndirectCallFixup::source_incomplete))
            has_unknown_exit = true;
        } else if (!is_program_exit) {
          if (auto branch_delta = term->branch_offset_bytes()) {
            const int64_t target = static_cast<int64_t>(block->end_offset()) + *branch_delta;
            if (target < 0 ||
                block_by_offset.find(static_cast<uint64_t>(target)) == block_by_offset.end())
              has_unknown_exit = true;
          }
          if (!is_unconditional_branch(*term) && (flags & INDIRECT_BRANCH) == 0 &&
              block_by_offset.find(block->end_offset()) == block_by_offset.end())
            has_unknown_exit = true;
        }

        if (block->successors().empty() && !is_program_exit)
          has_unknown_exit = true;
        for (BasicBlock *successor : block->successors())
          push_within_function(successor, point.terminal_return_sreg);
      }

      return has_unknown_exit ? CallReturnClassification::Unknown
                              : CallReturnClassification::NonReturning;
    };

    // Call sites form a small interprocedural graph. Resolve leaf callees first,
    // then repeat against a stable snapshot so nested tail transfers cannot make
    // classification depend on source or fixup order. Cycles without a positive
    // return or non-return proof remain conservative Unknown sites.
    size_t num_targets = 0;
    for (const DeferredCallSite &site : deferred_calls)
      num_targets += site.targets.size();
    const size_t max_rounds = num_targets + 2;
    bool converged = false;
    for (size_t round = 0; round < max_rounds; ++round) {
      std::vector<std::vector<CallReturnClassification>> next = classifications;
      bool changed = false;
      for (size_t site_index = 0; site_index < deferred_calls.size(); ++site_index) {
        const DeferredCallSite &site = deferred_calls[site_index];
        for (size_t target_index = 0; target_index < site.targets.size(); ++target_index) {
          BasicBlock *target = site.targets[target_index].target;
          if (target == nullptr)
            continue;
          const CallReturnClassification classification =
              classify_function(*target, site.return_sreg, classifications);
          if (classification != next[site_index][target_index]) {
            next[site_index][target_index] = classification;
            changed = true;
          }
        }
      }
      classifications = std::move(next);
      if (!changed) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      // A pathological non-monotone call graph must lose precision, not hang
      // or retain a potentially stale NonReturning classification.
      for (auto &site_classifications : classifications)
        std::ranges::fill(site_classifications, CallReturnClassification::Unknown);
    }

    for (size_t site_index = 0; site_index < deferred_calls.size(); ++site_index) {
      const DeferredCallSite &site = deferred_calls[site_index];

      bool all_targets_nonreturning = !site.target_set_incomplete;
      bool continuation_is_target = false;
      for (size_t target_index = 0; target_index < site.targets.size(); ++target_index) {
        const DeferredCallTarget &target = site.targets[target_index];
        const CallReturnClassification classification = classifications[site_index][target_index];
        all_targets_nonreturning &= classification == CallReturnClassification::NonReturning;
        continuation_is_target |= target.target == site.continuation;

        if (classification == CallReturnClassification::Returning) {
          site.source->add_call_edge(CallEdge{.kind = target.kind,
                                              .callee = target.target,
                                              .continuation = site.continuation,
                                              .source_call_offset = target.source_call_offset,
                                              .return_sreg = site.return_sreg});
        } else {
          // Proven tail targets and unknown callees both remain reachable.
          // Unknown callees also conservatively keep the syntactic continuation.
          site.source->add_successor(*target.target);
        }
      }

      if (all_targets_nonreturning && !continuation_is_target) {
        // Every finite target is proven to end without returning through this
        // call site's destination pair. Drop only this dead fallthrough; mixed
        // and unknown target sets retain the continuation conservatively.
        (void)site.source->remove_successor(*site.continuation);
      }
    }

    for (auto &block : section_blocks)
      blocks.push_back(std::move(block));
  }

  return blocks;
}

std::vector<std::unique_ptr<BasicBlock>>
BasicBlock::build_reachable(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                            std::span<const uint64_t> entry_offsets,
                            std::span<const uint64_t> entry_sizes, uint32_t wavefront_size) {
  if (entry_offsets.empty())
    return {};
  if (!entry_sizes.empty() && entry_sizes.size() != entry_offsets.size())
    throw util::InvalidInst("entry size count does not match entry count", "Invalid CFG: ");

  std::vector<CodeRange> reachable_ranges;
  std::vector<IndirectCallFixup> retained_indirect_targets;

  for (const auto *sec : co.text_sections()) {
    const auto text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(sec->data()), sec->size());
    const uint64_t section_end = text.size();
    if (section_end == 0)
      continue;

    std::vector<std::pair<uint64_t, uint64_t>> entry_ranges;
    for (size_t i = 0; i < entry_sizes.size(); ++i) {
      if (entry_sizes[i] == 0 || entry_offsets[i] >= section_end)
        continue;
      const uint64_t available = section_end - entry_offsets[i];
      entry_ranges.emplace_back(entry_offsets[i],
                                entry_offsets[i] + std::min(entry_sizes[i], available));
    }
    const auto reachable_end = [&](uint64_t offset) {
      for (const auto &[begin, end] : entry_ranges) {
        if (offset >= begin && offset < end)
          return end;
      }
      return section_end;
    };

    std::map<uint64_t, std::unique_ptr<Instruction>> decoded;
    std::set<uint64_t> leaders;
    std::unordered_set<uint64_t> enqueued;
    std::vector<uint64_t> worklist;

    auto enqueue = [&](uint64_t offset) {
      if (offset >= section_end)
        return;
      if (offset % sizeof(uint32_t) != 0)
        throw util::InvalidInst("unaligned branch target", "Invalid CFG: ");
      leaders.insert(offset);
      if (enqueued.insert(offset).second)
        worklist.push_back(offset);
    };

    for (uint64_t entry : entry_offsets)
      enqueue(entry);

    std::vector<IndirectCallFixup> recovered_indirect_targets;
    size_t work_index = 0;
    while (true) {
      while (work_index < worklist.size()) {
        uint64_t offset = worklist[work_index++];
        const uint64_t decode_end = reachable_end(offset);
        while (offset < decode_end) {
          if (offset % sizeof(uint32_t) != 0)
            throw util::InvalidInst("unaligned instruction offset", "Invalid CFG: ");

          auto decoded_it = decoded.find(offset);
          if (decoded_it == decoded.end()) {
            // Decode from a bounded local window. Zero padding preserves the
            // decoder's lookahead contract at a metadata-backed range boundary.
            std::array<uint32_t, Decoder::kMaximumInstructionWords> window{};
            const size_t available =
                static_cast<size_t>(std::min<uint64_t>(sizeof(window), decode_end - offset));
            std::memcpy(window.data(), text.data() + offset, available);
            std::unique_ptr<Instruction> inst(decoder.decode(window.data(), offset));
            if (!inst || inst->size() <= 0)
              throw util::InvalidInst("zero-sized instruction", "Invalid CFG: ");
            if (offset + static_cast<uint64_t>(inst->size()) > decode_end)
              throw util::InvalidInst("truncated instruction", "Invalid CFG: ");
            decoded_it = decoded.emplace(offset, std::move(inst)).first;
          }

          const Instruction &inst = *decoded_it->second;
          const uint64_t next_offset = offset + static_cast<uint64_t>(inst.size());
          const auto branch_delta = inst.branch_offset_bytes();
          assert((!(inst.flags() & (BRANCH | COND_BRANCH)) || branch_delta.has_value()) &&
                 "direct branch is missing branch_offset_bytes()");

          if (branch_delta) {
            const int64_t target =
                static_cast<int64_t>(next_offset) + static_cast<int64_t>(*branch_delta);
            if (target >= 0 && static_cast<uint64_t>(target) < section_end)
              enqueue(static_cast<uint64_t>(target));
          }

          if (is_block_terminator(inst)) {
            if (!has_no_static_successor(inst) && !is_unconditional_branch(inst) &&
                next_offset < decode_end)
              enqueue(next_offset);
            break;
          }
          if (next_offset >= decode_end)
            break;
          if (leaders.contains(next_offset)) {
            enqueue(next_offset);
            break;
          }
          offset = next_offset;
        }
      }

      std::vector<const Instruction *> decoded_insts;
      decoded_insts.reserve(decoded.size());
      std::vector<uint64_t> discovery_leaders(entry_offsets.begin(), entry_offsets.end());
      uint64_t previous_end = 0;
      bool first = true;
      for (const auto &[offset, inst] : decoded) {
        decoded_insts.push_back(inst.get());
        if (first || offset != previous_end)
          discovery_leaders.push_back(offset);
        previous_end = offset + static_cast<uint64_t>(inst->size());
        first = false;
      }
      discovery_leaders.insert(discovery_leaders.end(), leaders.begin(), leaders.end());
      std::ranges::sort(discovery_leaders);
      discovery_leaders.erase(std::ranges::unique(discovery_leaders).begin(),
                              discovery_leaders.end());

      const auto newly_recovered = discover_indirect_branch_edges(
          std::span<const Instruction *const>(decoded_insts.data(), decoded_insts.size()), text,
          arch, discovery_leaders, wavefront_size, entry_offsets);

      const auto same_fixup = [](const IndirectCallFixup &left, const IndirectCallFixup &right) {
        return left.source_call_offset == right.source_call_offset &&
               left.source_target_offset == right.source_target_offset &&
               left.source_call_selector == right.source_call_selector &&
               left.source_call_carrier == right.source_call_carrier &&
               left.source_return_selector == right.source_return_selector;
      };
      // Reachability grows monotonically, while a closed-target proof can only
      // weaken as newly decoded predecessors join a consumer.
      for (IndirectCallFixup &existing : recovered_indirect_targets) {
        if (std::ranges::none_of(newly_recovered, [&](const IndirectCallFixup &candidate) {
              return same_fixup(existing, candidate);
            })) {
          existing.source_incomplete = true;
          existing.source_targets_exhaustive = false;
        }
      }
      for (const IndirectCallFixup &fixup : newly_recovered) {
        const auto duplicate = std::ranges::find_if(
            recovered_indirect_targets,
            [&](const IndirectCallFixup &existing) { return same_fixup(existing, fixup); });
        if (duplicate == recovered_indirect_targets.end()) {
          recovered_indirect_targets.push_back(fixup);
        } else {
          duplicate->source_incomplete |= fixup.source_incomplete;
          duplicate->source_targets_exhaustive &= fixup.source_targets_exhaustive;
        }
        leaders.insert(fixup.source_call_offset);
        enqueue(fixup.source_target_offset);
      }

      if (work_index == worklist.size())
        break;
    }

    // Feed the exact decoded islands back through the normal merged CFG
    // finalizer. This keeps one implementation of block splitting, static
    // successor diagnostics, and nested call/return classification.
    auto decoded_it = decoded.begin();
    while (decoded_it != decoded.end()) {
      const uint64_t range_begin = decoded_it->first;
      uint64_t range_end = range_begin + static_cast<uint64_t>(decoded_it->second->size());
      ++decoded_it;
      while (decoded_it != decoded.end() && decoded_it->first == range_end) {
        range_end += static_cast<uint64_t>(decoded_it->second->size());
        ++decoded_it;
      }
      reachable_ranges.push_back({.start_offset = range_begin, .size = range_end - range_begin});
    }
    retained_indirect_targets.insert(retained_indirect_targets.end(),
                                     recovered_indirect_targets.begin(),
                                     recovered_indirect_targets.end());
  }

  return build_impl(co, decoder, arch, entry_offsets, ExternalEntryPolicy::ExplicitOnly,
                    reachable_ranges, retained_indirect_targets);
}

} // namespace rocjitsu
