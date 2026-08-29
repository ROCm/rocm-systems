// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_barrier_move_proof.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <ranges>
#include <string_view>
#include <unordered_set>

namespace rocjitsu {
namespace {

[[nodiscard]] bool block_reaches(const BasicBlock *begin, const BasicBlock *target,
                                 std::unordered_set<const BasicBlock *> &visited) {
  if (begin == target)
    return true;
  if (begin == nullptr || !visited.insert(begin).second)
    return false;
  return std::ranges::any_of(begin->successors(), [&](const BasicBlock *successor) {
    return block_reaches(successor, target, visited);
  });
}

[[nodiscard]] bool is_matching_exec_restore(const Instruction &restore,
                                            const Instruction &saveexec) {
  if (restore.mnemonic() != std::string_view("s_mov_b64") ||
      saveexec.mnemonic().find("_saveexec_") == std::string_view::npos ||
      restore.raw_encoding() == nullptr || saveexec.raw_encoding() == nullptr) {
    return false;
  }
  const uint32_t restore_word = restore.raw_encoding()[0];
  const uint32_t save_word = saveexec.raw_encoding()[0];
  constexpr uint32_t kExecLo = 126;
  const uint32_t restore_destination = (restore_word >> 16u) & 0x7fu;
  const uint32_t restore_source = restore_word & 0xffu;
  const uint32_t saved_destination = (save_word >> 16u) & 0x7fu;
  return restore_destination == kExecLo && restore_source == saved_destination;
}

[[nodiscard]] bool is_matching_cmpx_exec_restore(const Instruction &restore,
                                                 const Instruction &save) {
  if (restore.mnemonic() != std::string_view("s_or_b32") ||
      save.mnemonic() != std::string_view("s_mov_b32") || restore.raw_encoding() == nullptr ||
      save.raw_encoding() == nullptr) {
    return false;
  }
  const uint32_t restore_word = restore.raw_encoding()[0];
  const uint32_t save_word = save.raw_encoding()[0];
  constexpr uint32_t kExecLo = 126;
  const uint32_t restore_destination = (restore_word >> 16u) & 0x7fu;
  const uint32_t restore_source0 = restore_word & 0xffu;
  const uint32_t restore_source1 = (restore_word >> 8u) & 0xffu;
  const uint32_t saved_destination = (save_word >> 16u) & 0x7fu;
  const uint32_t saved_source = save_word & 0xffu;
  return restore_destination == kExecLo && restore_source0 == kExecLo &&
         restore_source1 == saved_destination && saved_source == kExecLo;
}

} // namespace

std::vector<std::unique_ptr<BasicBlock>>
build_original_proof_basic_blocks(const AmdGpuCodeObject &code_object, Decoder &decoder,
                                  rj_code_arch_t arch) {
  std::vector<uint64_t> leaders;
  leaders.reserve(code_object.kernels().size() + code_object.functions().size());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.has_text_range)
      leaders.push_back(kernel.entry_text_offset);
  }
  for (const AmdGpuFunctionInfo &function : code_object.functions())
    leaders.push_back(function.entry_text_offset);
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());

  std::vector<BasicBlock::CodeRange> code_ranges;
  code_ranges.reserve(code_object.functions().size());
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  return BasicBlock::build(code_object, decoder, arch, leaders, code_ranges);
}

std::optional<StructuredExecDiamondProof>
prove_structured_exec_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              uint32_t source_index, uint32_t destination_index,
                              uint64_t barrier_begin) {
  if (source_index >= blocks.size() || destination_index >= blocks.size() ||
      source_index == destination_index) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[source_index].get();
  BasicBlock *destination = blocks[destination_index].get();
  if (source == nullptr || destination == nullptr || source->num_instructions() == 0u ||
      destination->successors().size() != 1u || destination->successors().front() != source ||
      !destination->call_edges().empty()) {
    return std::nullopt;
  }
  const Instruction &restore = *source->instructions().begin();
  if (restore.src_loc() != source->start_offset() ||
      restore.size() != static_cast<int>(sizeof(uint32_t)) ||
      barrier_begin != source->start_offset() + sizeof(uint32_t)) {
    return std::nullopt;
  }

  std::vector<BasicBlock *> guards;
  for (BasicBlock *candidate : destination->predecessors()) {
    if (candidate != nullptr &&
        std::ranges::find(source->predecessors(), candidate) != source->predecessors().end()) {
      guards.push_back(candidate);
    }
  }
  if (guards.size() != 1u || source->predecessors().size() != 2u ||
      destination->predecessors().size() != 1u) {
    return std::nullopt;
  }
  BasicBlock *guard = guards.front();
  if (guard->successors().size() != 2u || !guard->call_edges().empty() ||
      std::ranges::find(guard->successors(), source) == guard->successors().end() ||
      std::ranges::find(guard->successors(), destination) == guard->successors().end() ||
      guard->num_instructions() < 2u) {
    return std::nullopt;
  }
  const Instruction *terminator = guard->terminator();
  if (terminator == nullptr || (terminator->mnemonic() != std::string_view("s_cbranch_execz") &&
                                terminator->mnemonic() != std::string_view("s_cbranch_execnz"))) {
    return std::nullopt;
  }
  const Instruction *before_narrow = nullptr;
  const Instruction *narrow = nullptr;
  for (const Instruction &instruction : guard->instructions()) {
    if (&instruction == terminator)
      break;
    before_narrow = narrow;
    narrow = &instruction;
  }
  if (narrow == nullptr || narrow->src_loc() + narrow->size() != terminator->src_loc()) {
    return std::nullopt;
  }
  const bool matched_saveexec = is_matching_exec_restore(restore, *narrow);
  const bool matched_cmpx = narrow->mnemonic().starts_with("v_cmpx_") && before_narrow != nullptr &&
                            before_narrow->src_loc() + before_narrow->size() == narrow->src_loc() &&
                            is_matching_cmpx_exec_restore(restore, *before_narrow);
  if (!matched_saveexec && !matched_cmpx) {
    return std::nullopt;
  }

  std::unordered_set<const BasicBlock *> visited;
  if (block_reaches(source, guard, visited))
    return std::nullopt;
  visited.clear();
  if (block_reaches(source, destination, visited))
    return std::nullopt;

  const auto guard_it = std::ranges::find(blocks, guard, &std::unique_ptr<BasicBlock>::get);
  if (guard_it == blocks.end())
    return std::nullopt;
  return StructuredExecDiamondProof{
      .guard_block_index = static_cast<uint32_t>(guard_it - blocks.begin()),
      .destination_block_index = destination_index,
      .source_block_index = source_index,
      .guard_offset = terminator->src_loc(),
      .destination_offset = destination->start_offset(),
      .source_offset = source->start_offset(),
  };
}

std::optional<StructuredExecDiamondProof>
prove_completing_structured_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                    uint32_t source_index, uint32_t destination_index,
                                    uint64_t barrier_begin, uint64_t destination_offset) {
  if (source_index >= blocks.size() || destination_index >= blocks.size() ||
      source_index == destination_index) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[source_index].get();
  BasicBlock *guard = blocks[destination_index].get();
  if (source == nullptr || guard == nullptr || barrier_begin != source->start_offset() ||
      destination_offset < guard->start_offset() || destination_offset >= guard->end_offset() ||
      source->predecessors().size() != 2u || guard->successors().size() != 2u ||
      !source->call_edges().empty() || !guard->call_edges().empty()) {
    return std::nullopt;
  }
  const Instruction *terminator = guard->terminator();
  if (terminator == nullptr || !(terminator->flags() & COND_BRANCH) ||
      terminator->branch_offset_bytes() == 0) {
    return std::nullopt;
  }

  std::unordered_set<const BasicBlock *> expected_predecessors;
  for (BasicBlock *arm : guard->successors()) {
    if (arm == nullptr || arm == source || arm == guard || !arm->call_edges().empty() ||
        arm->predecessors().size() != 1u || arm->predecessors().front() != guard ||
        arm->successors().size() != 1u || arm->successors().front() != source) {
      return std::nullopt;
    }
    expected_predecessors.insert(arm);
  }
  if (expected_predecessors.size() != 2u ||
      !std::ranges::all_of(source->predecessors(), [&](const BasicBlock *predecessor) {
        return expected_predecessors.contains(predecessor);
      })) {
    return std::nullopt;
  }
  std::unordered_set<const BasicBlock *> visited;
  if (block_reaches(source, guard, visited))
    return std::nullopt;

  return StructuredExecDiamondProof{
      .guard_block_index = destination_index,
      .destination_block_index = destination_index,
      .source_block_index = source_index,
      .guard_offset = terminator->src_loc(),
      .destination_offset = destination_offset,
      .source_offset = source->start_offset(),
  };
}

} // namespace rocjitsu
