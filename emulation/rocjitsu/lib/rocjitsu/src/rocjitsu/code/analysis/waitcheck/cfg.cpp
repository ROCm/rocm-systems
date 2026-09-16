// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/cfg.h"

#include "rocjitsu/code/analysis/waitcheck/transfer.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace rocjitsu {
namespace {
using namespace waitcheck_detail;
using Ops = WaitcheckStateOps;

void merge_into(MemoryTransferState &dst, const MemoryTransferState &src) {
  Ops::merge_into(dst.pending, src.pending);
  dst.local_ready &= src.local_ready;
}

util::FailureOr<std::optional<WaitcheckStreamStop>>
transfer_block(BasicBlock &block, MemoryTransferState &state, rj_code_arch_t arch,
               WaitcheckStreamOptions options, WaitcheckStreamReport *report) {
  for (const Instruction &inst : block.instructions()) {
    auto stop = transfer_memory_instruction(state, inst, arch, options, true, report);
    if (stop.failed() || stop.value())
      return stop;
  }
  return std::optional<WaitcheckStreamStop>{};
}
} // namespace

util::FailureOr<WaitcheckCfgReport>
analyze_waitcheck_cfg(const CodeObject &object, rj_code_arch_t arch,
                      std::span<const uint64_t> entries, WaitcheckCfgOptions options,
                      const util::DiagnosticEmitter &emit_error) {
  if (validate_memory_options(arch, options.entry_modes, emit_error).failed())
    return util::Result::failure();
  if (options.max_block_visits == 0)
    return emit_error.emit() << "waitcheck block visit budget must be positive";
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return emit_error.emit() << "missing waitcheck ISA decoder";
  auto built = BasicBlock::build_cfg(
      object, *decoder, arch, {.entries = entries, .permitted_ranges = options.permitted_ranges},
      emit_error);
  if (built.failed())
    return util::Result::failure();
  const auto &blocks = built.value();
  const size_t num_blocks = blocks.size();
  std::unordered_map<const BasicBlock *, size_t> block_index;
  for (size_t i = 0; i < num_blocks; ++i)
    block_index.emplace(blocks[i].get(), i);
  std::vector<MemoryTransferState> in(num_blocks), out(num_blocks);
  std::vector<uint8_t> initialized(num_blocks), propagates(num_blocks), external(num_blocks),
      in_worklist(num_blocks);
  std::deque<size_t> worklist;
  auto enqueue = [&](size_t idx) {
    if (!in_worklist[idx]) {
      in_worklist[idx] = 1;
      worklist.push_back(idx);
    }
  };
  for (size_t i = 0; i < num_blocks; ++i) {
    external[i] = std::ranges::find(entries, blocks[i]->start_offset()) != entries.end();
    if (external[i])
      enqueue(i);
  }
  MemoryTransferState entry;
  entry.pending.expert_scheduling.enabled = options.entry_modes.expert_scheduling;
  size_t visits = 0;
  while (!worklist.empty()) {
    const size_t idx = worklist.front();
    worklist.pop_front();
    in_worklist[idx] = 0;
    if (visits++ == options.max_block_visits)
      return emit_error.emit()
             << "waitcheck CFG did not converge within block visit budget at byte "
             << blocks[idx]->start_offset();
    MemoryTransferState input = entry;
    bool has_input = external[idx];
    for (const BasicBlock *pred : blocks[idx]->predecessors()) {
      const size_t predecessor = block_index.find(pred)->second;
      if (!initialized[predecessor] || !propagates[predecessor])
        continue;
      if (has_input)
        merge_into(input, out[predecessor]);
      else
        input = out[predecessor];
      has_input = true;
    }
    if (!has_input)
      continue;
    auto output = input;
    auto stop = transfer_block(*blocks[idx], output, arch, options.entry_modes, nullptr);
    if (stop.failed())
      return emit_error.emit() << "waitcheck block transfer failed at byte "
                               << blocks[idx]->start_offset();
    const bool changed = !initialized[idx] || !(output == out[idx]);
    in[idx] = std::move(input);
    out[idx] = std::move(output);
    initialized[idx] = 1;
    propagates[idx] = !stop.value();
    if (changed && propagates[idx]) {
      for (const BasicBlock *succ : blocks[idx]->successors())
        enqueue(block_index.find(succ)->second);
    }
  }

  WaitcheckCfgReport report;
  WaitcheckStreamReport diagnostics;
  for (size_t i = 0; i < num_blocks; ++i) {
    if (!initialized[i])
      continue;
    ++report.blocks_analyzed;
    auto state = in[i];
    auto stop = transfer_block(*blocks[i], state, arch, options.entry_modes, &diagnostics);
    if (stop.failed())
      return emit_error.emit() << "waitcheck diagnostic transfer failed at byte "
                               << blocks[i]->start_offset();
    if (stop.value()) {
      report.incomplete.push_back(std::move(*stop.value()));
    } else if (blocks[i]->successor_issue() != BasicBlock::SuccessorIssue::None ||
               !blocks[i]->call_edges().empty()) {
      const auto *term = blocks[i]->terminator();
      report.incomplete.push_back({term ? term->src_loc() : blocks[i]->end_offset(),
                                   term ? term->disassemble() : std::string{},
                                   "CFG has missing or unsupported control flow"});
    }
  }
  report.diagnostics = std::move(diagnostics.diagnostics);
  report.instructions_analyzed = diagnostics.instructions_analyzed;
  return report;
}
} // namespace rocjitsu
