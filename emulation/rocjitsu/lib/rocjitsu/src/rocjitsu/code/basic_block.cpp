// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"

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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

uint32_t first_word(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  return raw == nullptr ? 0 : raw[0];
}

bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  if (inst.size() != sizeof(uint32_t) ||
      (inst.mnemonic() != "s_setpc_b64" && inst.mnemonic() != "s_set_pc_i64"))
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

struct DeferredIndirectCall {
  BasicBlock *source = nullptr;
  BasicBlock *target = nullptr;
  BasicBlock *continuation = nullptr;
  uint64_t source_call_offset = 0;
  uint16_t return_sreg = 0;
};

struct DeferredDirectCall {
  BasicBlock *source = nullptr;
  BasicBlock *target = nullptr;
  BasicBlock *continuation = nullptr;
  uint64_t source_call_offset = 0;
  uint16_t return_sreg = 0;
};

struct StaticSuccessorResolution {
  BasicBlock *target = nullptr;
  BasicBlock *fallthrough = nullptr;
  std::optional<uint16_t> direct_call_return_sreg;
  bool expects_fallthrough = false;
  BasicBlock::StaticSuccessorIssue issue = BasicBlock::StaticSuccessorIssue::None;
};

[[nodiscard]] StaticSuccessorResolution
resolve_static_successors(const BasicBlock &block,
                          const std::unordered_map<uint64_t, BasicBlock *> &block_by_offset) {
  StaticSuccessorResolution result;
  const Instruction *term = block.terminator();
  assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");

  const auto branch_delta = term->branch_offset_bytes();
  assert((!(term->flags() & (BRANCH | COND_BRANCH)) || branch_delta.has_value()) &&
         "direct branch is missing branch_offset_bytes()");

  if (auto it = block_by_offset.find(block.end_offset()); it != block_by_offset.end())
    result.fallthrough = it->second;
  result.direct_call_return_sreg = s_call_sdst(*term, first_word(*term));
  const bool is_call = (term->flags() & INDIRECT_CALL) != 0;
  result.expects_fallthrough = !is_unconditional_branch(*term);

  if (branch_delta) {
    // BasicBlock::end_offset() is the next instruction address for the
    // terminator, which is the base used by AMDGPU direct branch labels.
    const int64_t target =
        static_cast<int64_t>(block.end_offset()) + static_cast<int64_t>(*branch_delta);
    if (auto it = target >= 0 ? block_by_offset.find(static_cast<uint64_t>(target))
                              : block_by_offset.end();
        it != block_by_offset.end()) {
      result.target = it->second;
    } else {
      result.issue = is_call ? BasicBlock::StaticSuccessorIssue::MissingCallTarget
                             : BasicBlock::StaticSuccessorIssue::MissingBranchTarget;
    }
  }

  if (result.expects_fallthrough && result.fallthrough == nullptr &&
      result.issue == BasicBlock::StaticSuccessorIssue::None) {
    result.issue = is_call ? BasicBlock::StaticSuccessorIssue::MissingCallContinuation
                           : BasicBlock::StaticSuccessorIssue::MissingFallthrough;
  }
  return result;
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

std::vector<std::unique_ptr<BasicBlock>> BasicBlock::build(const CodeObject &co, Decoder &decoder,
                                                           rj_code_arch_t arch,
                                                           std::span<const uint64_t> extra_leaders,
                                                           std::span<const CodeRange> code_ranges) {
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
        if (range.start_offset >= sec->size() || range.size == 0)
          continue;
        if (range.start_offset % sizeof(uint32_t) != 0 || range.size % sizeof(uint32_t) != 0) {
          throw std::runtime_error("Declared code range is not dword aligned");
        }
        const uint64_t available = sec->size() - range.start_offset;
        section_ranges.push_back(
            {.start_offset = range.start_offset, .size = std::min(range.size, available)});
      }
      std::ranges::sort(section_ranges, {}, &CodeRange::start_offset);
      std::vector<CodeRange> merged_ranges;
      for (const CodeRange &range : section_ranges) {
        if (merged_ranges.empty() ||
            merged_ranges.back().start_offset + merged_ranges.back().size <= range.start_offset) {
          merged_ranges.push_back(range);
          continue;
        }
        CodeRange &merged = merged_ranges.back();
        const uint64_t merged_end = merged.start_offset + merged.size;
        const uint64_t range_end = range.start_offset + range.size;
        merged.size = std::max(merged_end, range_end) - merged.start_offset;
      }
      section_ranges = std::move(merged_ranges);
    }

    for (const CodeRange &range : section_ranges) {
      uint64_t byte_offset = range.start_offset;
      const uint64_t range_end = range.start_offset + range.size;
      while (byte_offset < range_end) {
        // Kernel symbols may include zero alignment padding after their
        // terminal s_endpgm. Do not ask the ISA decoder to interpret that
        // non-executable tail as instructions.
        if (!decoded.empty() && decoded.back()->mnemonic() == std::string_view("s_endpgm")) {
          const auto *tail_begin = reinterpret_cast<const uint8_t *>(sec->data()) + byte_offset;
          const auto tail = std::span<const uint8_t>(tail_begin, range_end - byte_offset);
          if (std::ranges::all_of(tail, [](uint8_t byte) { return byte == 0u; }))
            break;
        }
        const size_t pc = static_cast<size_t>(byte_offset / sizeof(uint32_t));
        Instruction *raw_inst = nullptr;
        try {
          raw_inst = decoder.decode(&inst_data[pc], byte_offset);
        } catch (const std::exception &error) {
          throw std::runtime_error("Cannot decode basic-block instruction at text offset " +
                                   std::to_string(byte_offset) + ": " + error.what());
        }
        std::unique_ptr<Instruction> inst(raw_inst);
        const uint32_t inst_size_bytes = static_cast<uint32_t>(inst->size());
        if (inst_size_bytes == 0 || inst_size_bytes > range_end - byte_offset)
          throw std::runtime_error("Instruction extends past declared code range");

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
    std::vector<IndirectCallFixup> recovered_indirect_targets =
        discover_indirect_branch_edges(decoded_span, text, arch, extra_leaders);

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

        const bool discontinuity = i < decoded.size() && decoded[i]->src_loc() != next_offset;
        if (terminates || discontinuity ||
            (i < decoded.size() && leaders.contains(decoded[i]->src_loc())))
          break;
      }
      section_blocks.push_back(std::move(current));
    }

    std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
    block_by_offset.reserve(section_blocks.size());
    for (auto &block : section_blocks)
      block_by_offset.emplace(block->start_offset(), block.get());

    std::unordered_set<uint64_t> kernel_entry_offsets(extra_leaders.begin(), extra_leaders.end());
    auto function_returns_to_sreg = [&](BasicBlock &callee, uint16_t return_sreg) {
      std::vector<BasicBlock *> stack{&callee};
      std::unordered_set<BasicBlock *> visited;

      while (!stack.empty()) {
        BasicBlock *block = stack.back();
        stack.pop_back();
        if (block == nullptr || !visited.insert(block).second)
          continue;

        const Instruction *term = block->terminator();
        if (term == nullptr)
          continue;

        if (s_setpc_from_sreg(*term, first_word(*term), return_sreg))
          return true;

        // Call validation runs after ordinary direct CFG edges and recovered
        // non-call setpc edges have been installed in successors(). Follow that
        // graph instead of re-deriving direct edges here; shared helpers often
        // branch through a recovered setpc before reaching the setpc return.
        // Kernel entries remain scope boundaries, except when the callee itself
        // is a kernel entry. That keeps a helper walk from proving a return by
        // wandering into an unrelated kernel body.
        for (BasicBlock *succ : block->successors()) {
          if (succ == nullptr)
            continue;
          if (kernel_entry_offsets.contains(succ->start_offset()) && succ != &callee)
            continue;
          stack.push_back(succ);
        }
      }

      return false;
    };

    std::vector<DeferredIndirectCall> deferred_indirect_calls;
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
          deferred_indirect_calls.push_back({.source = source,
                                             .target = target,
                                             .continuation = continuation,
                                             .source_call_offset = fixup.source_call_offset,
                                             .return_sreg = fixup.source_return_sreg});
        } else {
          // Non-call recovered setpc targets are ordinary local CFG edges. If a
          // swappc has no statically-known continuation, keep the old
          // conservative reachability edge instead of pretending it has
          // call/return semantics.
          source->add_successor(*target);
        }
      }
    }

    std::vector<DeferredDirectCall> deferred_direct_calls;
    for (size_t i = 0; i < section_blocks.size(); ++i) {
      auto &block = *section_blocks[i];
      const Instruction *term = block.terminator();
      if (term == nullptr || has_no_static_successor(*term))
        continue;

      const StaticSuccessorResolution resolution =
          resolve_static_successors(block, block_by_offset);
      block.note_static_successor_issue(resolution.issue);
      if (resolution.target != nullptr) {
        if (resolution.direct_call_return_sreg && resolution.fallthrough != nullptr) {
          // Like recovered swappc, direct s_call validation needs the callee's
          // internal CFG to be complete before we decide whether the target is
          // a returning helper or an ordinary reachable branch target.
          deferred_direct_calls.push_back({.source = &block,
                                           .target = resolution.target,
                                           .continuation = resolution.fallthrough,
                                           .source_call_offset = term->src_loc(),
                                           .return_sreg = *resolution.direct_call_return_sreg});
        } else {
          block.add_successor(*resolution.target);
        }
      }

      if (resolution.expects_fallthrough && resolution.fallthrough != nullptr)
        block.add_successor(*resolution.fallthrough);
    }

    for (const DeferredIndirectCall &call : deferred_indirect_calls) {
      if (call.source == nullptr || call.target == nullptr || call.continuation == nullptr)
        continue;
      if (function_returns_to_sreg(*call.target, call.return_sreg)) {
        call.source->add_call_edge(CallEdge{.kind = CallEdgeKind::IndirectSwapPc,
                                            .callee = call.target,
                                            .continuation = call.continuation,
                                            .source_call_offset = call.source_call_offset,
                                            .return_sreg = call.return_sreg});
      } else {
        // A statically recovered swappc target that does not return through the
        // swappc destination is just indirect control flow with a concrete
        // target. Model that conservatively as an ordinary CFG edge.
        call.source->add_successor(*call.target);
      }
    }

    for (const DeferredDirectCall &call : deferred_direct_calls) {
      if (call.source == nullptr || call.target == nullptr || call.continuation == nullptr)
        continue;
      if (function_returns_to_sreg(*call.target, call.return_sreg)) {
        call.source->add_call_edge(CallEdge{.kind = CallEdgeKind::DirectCall,
                                            .callee = call.target,
                                            .continuation = call.continuation,
                                            .source_call_offset = call.source_call_offset,
                                            .return_sreg = call.return_sreg});
      } else {
        call.source->add_successor(*call.target);
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
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  if (entry_offsets.empty())
    return blocks;
  if (!entry_sizes.empty() && entry_sizes.size() != entry_offsets.size())
    throw util::InvalidInst("entry size count does not match entry count", "Invalid CFG: ");

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
            // Decode from a small local window instead of copying the complete
            // .text section. AMDGPU instructions occupy at most three words;
            // zero padding preserves the decoder's established lookahead
            // contract at the end of a section.
            std::array<uint32_t, 3> window{};
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
               left.source_call_sreg == right.source_call_sreg &&
               left.source_return_sreg == right.source_return_sreg;
      };
      // The decoded graph only grows. If rediscovery no longer emits a
      // previously recovered source/target observation, retain the target for
      // relocation and reachability but invalidate the stale closed-target
      // proof. A newly decoded predecessor may have killed the tracked PC pair
      // completely, in which case there is no replacement fixup to merge.
      for (IndirectCallFixup &existing : recovered_indirect_targets) {
        if (std::ranges::none_of(newly_recovered, [&](const IndirectCallFixup &candidate) {
              return same_fixup(existing, candidate);
            })) {
          existing.source_targets_exhaustive = false;
        }
      }
      for (const IndirectCallFixup &fixup : newly_recovered) {
        const auto duplicate = std::ranges::find_if(
            recovered_indirect_targets,
            [&](const IndirectCallFixup &existing) { return same_fixup(existing, fixup); });
        if (duplicate == recovered_indirect_targets.end())
          recovered_indirect_targets.push_back(fixup);
        else
          // Recovery is iterative: newly decoded predecessors can only weaken
          // an earlier closed-target proof, never strengthen it.
          duplicate->source_targets_exhaustive =
              duplicate->source_targets_exhaustive && fixup.source_targets_exhaustive;
        leaders.insert(fixup.source_call_offset);
        enqueue(fixup.source_target_offset);
      }

      if (work_index == worklist.size())
        break;
    }

    std::vector<std::unique_ptr<BasicBlock>> section_blocks;
    for (auto it = decoded.begin(); it != decoded.end();) {
      auto current = std::make_unique<BasicBlock>(it->first);
      while (it != decoded.end()) {
        const uint64_t inst_offset = it->first;
        const uint64_t next_offset = inst_offset + static_cast<uint64_t>(it->second->size());
        const bool terminates = is_block_terminator(*it->second);
        current->add_instruction(std::move(it->second));
        ++it;
        if (terminates || it == decoded.end() || it->first != next_offset ||
            leaders.contains(next_offset))
          break;
      }
      section_blocks.push_back(std::move(current));
    }

    std::unordered_map<uint64_t, BasicBlock *> block_by_offset;
    for (auto &block : section_blocks)
      block_by_offset.emplace(block->start_offset(), block.get());

    std::unordered_set<uint64_t> kernel_entry_offsets(entry_offsets.begin(), entry_offsets.end());
    auto function_returns_to_sreg = [&](BasicBlock &callee, uint16_t return_sreg) {
      std::vector<BasicBlock *> stack{&callee};
      std::unordered_set<BasicBlock *> visited;
      while (!stack.empty()) {
        BasicBlock *block = stack.back();
        stack.pop_back();
        if (block == nullptr || !visited.insert(block).second)
          continue;
        const Instruction *term = block->terminator();
        if (term == nullptr)
          continue;
        if (s_setpc_from_sreg(*term, first_word(*term), return_sreg))
          return true;
        for (BasicBlock *succ : block->successors()) {
          if (succ == nullptr)
            continue;
          if (kernel_entry_offsets.contains(succ->start_offset()) && succ != &callee)
            continue;
          stack.push_back(succ);
        }
      }
      return false;
    };

    std::vector<DeferredIndirectCall> deferred_indirect_calls;
    for (const IndirectCallFixup &fixup : recovered_indirect_targets) {
      auto source_it = block_by_offset.find(fixup.source_call_offset);
      if (source_it == block_by_offset.end())
        continue;
      BasicBlock *source = source_it->second;
      source->add_static_indirect_call_fixup(fixup);
      auto target_it = block_by_offset.find(fixup.source_target_offset);
      if (target_it == block_by_offset.end())
        continue;
      BasicBlock *target = target_it->second;
      BasicBlock *continuation = nullptr;
      if (auto it = block_by_offset.find(source->end_offset()); it != block_by_offset.end())
        continuation = it->second;
      if (fixup.source_is_call && continuation != nullptr) {
        deferred_indirect_calls.push_back({.source = source,
                                           .target = target,
                                           .continuation = continuation,
                                           .source_call_offset = fixup.source_call_offset,
                                           .return_sreg = fixup.source_return_sreg});
      } else {
        source->add_successor(*target);
      }
    }

    std::vector<DeferredDirectCall> deferred_direct_calls;
    for (auto &block_ptr : section_blocks) {
      BasicBlock &block = *block_ptr;
      const Instruction *term = block.terminator();
      if (term == nullptr || has_no_static_successor(*term))
        continue;

      const StaticSuccessorResolution resolution =
          resolve_static_successors(block, block_by_offset);
      block.note_static_successor_issue(resolution.issue);
      if (resolution.target != nullptr) {
        if (resolution.direct_call_return_sreg && resolution.fallthrough != nullptr) {
          deferred_direct_calls.push_back({.source = &block,
                                           .target = resolution.target,
                                           .continuation = resolution.fallthrough,
                                           .source_call_offset = term->src_loc(),
                                           .return_sreg = *resolution.direct_call_return_sreg});
        } else {
          block.add_successor(*resolution.target);
        }
      }
      if (resolution.expects_fallthrough && resolution.fallthrough != nullptr)
        block.add_successor(*resolution.fallthrough);
    }

    for (const DeferredIndirectCall &call : deferred_indirect_calls) {
      if (call.source == nullptr || call.target == nullptr || call.continuation == nullptr)
        continue;
      if (function_returns_to_sreg(*call.target, call.return_sreg)) {
        call.source->add_call_edge(CallEdge{.kind = CallEdgeKind::IndirectSwapPc,
                                            .callee = call.target,
                                            .continuation = call.continuation,
                                            .source_call_offset = call.source_call_offset,
                                            .return_sreg = call.return_sreg});
      } else {
        call.source->add_successor(*call.target);
      }
    }
    for (const DeferredDirectCall &call : deferred_direct_calls) {
      if (call.source == nullptr || call.target == nullptr || call.continuation == nullptr)
        continue;
      if (function_returns_to_sreg(*call.target, call.return_sreg)) {
        call.source->add_call_edge(CallEdge{.kind = CallEdgeKind::DirectCall,
                                            .callee = call.target,
                                            .continuation = call.continuation,
                                            .source_call_offset = call.source_call_offset,
                                            .return_sreg = call.return_sreg});
      } else {
        call.source->add_successor(*call.target);
      }
    }

    for (auto &block : section_blocks)
      blocks.push_back(std::move(block));
  }
  return blocks;
}

} // namespace rocjitsu
