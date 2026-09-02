// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"

#include "rocjitsu/analysis/control_flow.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include "util/except.h"

#include <algorithm>
#include <cassert>
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

FailureOr<std::vector<std::unique_ptr<BasicBlock>>>
BasicBlock::build(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                  DecodeErrorEmitter emit_error, std::span<const uint64_t> extra_leaders,
                  ExternalEntryPolicy entry_policy, std::span<const uint64_t> extra_split_points) {
  std::vector<std::unique_ptr<BasicBlock>> blocks;

  for (const auto *sec : co.text_sections()) {
    const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
    std::size_t inst_data_size = sec->size() / sizeof(uint32_t);
    uint64_t pc = 0;
    uint64_t byte_offset = 0;

    std::vector<std::unique_ptr<Instruction>> decoded;

    while (pc < inst_data_size) {
      // gfx1250 code objects use zero-filled alignment between function bodies.
      // Zero is not an instruction; block construction below treats sequential
      // fallthrough into it as an implicit unreachable boundary.
      if (arch == ROCJITSU_CODE_ARCH_CDNA5 && inst_data[pc] == 0) {
        ++pc;
        byte_offset += sizeof(uint32_t);
        continue;
      }

      auto emit_at_offset = [&](std::string_view message) {
        emit_error.emit() << message << " at .text byte offset " << byte_offset;
      };
      const DecodeErrorEmitter decode_error =
          emit_error.ignores_messages() ? DecodeErrorEmitter{} : DecodeErrorEmitter(emit_at_offset);
      DecodeResult decode_result = decoder.decode(&inst_data[pc], byte_offset, decode_error);
      if (decode_result.failed())
        return Result::failure();
      std::unique_ptr<Instruction> inst = std::move(decode_result).value();
      uint32_t inst_size_bytes = static_cast<uint32_t>(inst->size());
      uint32_t inst_words = inst_size_bytes / sizeof(uint32_t);

      decoded.push_back(std::move(inst));
      pc += inst_words;
      byte_offset += inst_size_bytes;
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
    const auto *amdgpu_object = dynamic_cast<const AmdGpuCodeObject *>(&co);
    const rj_code_target_id_t concrete_target =
        amdgpu_object != nullptr ? amdgpu_object->target_id() : ROCJITSU_CODE_TARGET_INVALID;
    std::vector<IndirectCallFixup> recovered_indirect_targets =
        discover_indirect_branch_edges(decoded_span, text, arch, extra_leaders, entry_policy,
                                       &pc_address_builders, extra_split_points, concrete_target);

    std::set<uint64_t> leaders;
    leaders.insert(decoded.front()->src_loc());
    for (uint64_t leader : extra_leaders) {
      if (leader < section_end)
        leaders.insert(leader);
    }
    // Split points shape the block graph only. They are deliberately absent from the external-entry
    // set built below, so a helper named by a function symbol keeps the caller facts it is entered
    // with.
    for (uint64_t split : extra_split_points) {
      if (split < section_end)
        leaders.insert(split);
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

        const bool decode_gap =
            i >= decoded.size() ? next_offset < section_end : decoded[i]->src_loc() != next_offset;
        const Instruction &last = *current->terminator();
        const bool can_fall_through = !is_program_path_terminator(last) &&
                                      !is_unconditional_branch(last) &&
                                      (last.flags() & INDIRECT_BRANCH) == 0;
        const bool reaches_gfx1250_zero = decode_gap && arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                                          next_offset < section_end &&
                                          inst_data[next_offset / sizeof(uint32_t)] == 0;
        // Running off the end of `.text` is the same boundary as running into padding: there is no
        // next instruction either way. Requiring padding to be present would make the result
        // depend on whether the linker happened to align the section, so an unterminated tail
        // would be translated verbatim in one build and given a terminator in the next.
        const bool reaches_section_end =
            arch == ROCJITSU_CODE_ARCH_CDNA5 && i >= decoded.size() && next_offset >= section_end;
        if (can_fall_through && (reaches_gfx1250_zero || reaches_section_end)) {
          current->has_terminator_ = true;
          current->has_implicit_terminator_ = true;
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
      }
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
          }
        }
      }

      const auto fallthrough_it = block_by_offset.find(block.end_offset());
      // Conditional branches and ordinary instructions may fall through; direct
      // unconditional branches do not.
      if (!is_unconditional_branch(*term) && fallthrough_it != block_by_offset.end())
        block.add_successor(*fallthrough_it->second);
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

        const uint64_t flags = term->flags();
        // An implicit terminator cuts the FALLTHROUGH edge only: the padding after this block is
        // not code, so control cannot continue past it. A branch terminator still has its taken
        // edge, and that edge's target must still be proven present. Treating the whole block as a
        // program exit would skip both missing-target checks below, and an unresolved taken target
        // would then leave has_unknown_exit false -- classifying the callee NonReturning and
        // deleting the caller's continuation.
        const bool has_branch_exit = term->branch_offset_bytes().has_value() ||
                                     (flags & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0;
        const bool is_program_exit = (block->has_implicit_terminator() && !has_branch_exit) ||
                                     is_program_path_terminator(*term);
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

} // namespace rocjitsu
