// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/local_helper.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/patch/consan/input.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>

namespace rocjitsu::consan {

std::span<BasicBlock *const> LocalHelper::predecessors(const BasicBlock &block) const {
  if (&block == callee)
    return {&caller, 1};
  if (&block == continuation)
    return {&callee, 1};
  return block.predecessors();
}
std::span<BasicBlock *const> LocalHelper::successors(const BasicBlock &block) const {
  if (&block == caller)
    return {&callee, 1};
  if (&block == callee)
    return {&continuation, 1};
  return block.successors();
}
bool LocalHelper::is_transfer(const Instruction &instruction) const {
  return caller && (&instruction == caller->terminator() || &instruction == callee->terminator());
}

FailureOr<LocalHelper> qualify_local_helper(const AmdGpuCodeObject &object,
                                            std::span<BasicBlock *const> blocks, uint32_t wave_size,
                                            InputError &error) {
  LocalHelper helper;
  for (BasicBlock *block : blocks) {
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      auto reject = [&](std::string message) {
        error = {InputErrorCode::UnsupportedFlow, call.source_call_offset, std::move(message)};
        return Result::failure();
      };
      if (helper.caller || call.kind != BasicBlock::CallEdgeKind::DirectCall || !call.callee ||
          !call.continuation || call.callee == block || call.callee == call.continuation ||
          call.continuation == block || !block->terminator() ||
          call.source_call_offset != block->terminator()->src_loc() ||
          block->successors().size() != 1 || block->successors().front() != call.continuation ||
          call.continuation->predecessors().size() != 1 ||
          call.continuation->predecessors().front() != block ||
          !call.callee->successors().empty() || !call.callee->predecessors().empty() ||
          !call.callee->call_edges().empty() || !call.callee->terminator())
        return reject("LDS helper requires one direct call and a single local return");
      const Section &section = *object.text_sections().front();
      const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(section.data()),
                                          section.size());
      helper.edges = scoped_call_liveness_edges(blocks, text);
      const auto returns = scoped_call_return_offsets(blocks, text);
      const auto has_edge = [&](const BasicBlock *from, const BasicBlock *to) {
        return std::ranges::any_of(
            helper.edges, [&](const auto &edge) { return edge.from == from && edge.to == to; });
      };
      if (helper.edges.size() != 2 || !has_edge(block, call.callee) ||
          !has_edge(call.callee, call.continuation) || returns.size() != 1 ||
          !returns.contains(call.callee->terminator()->src_loc()))
        return reject("LDS helper return does not match the calling SGPR pair and continuation");
      const auto name =
          local_text_function_name(object, call.callee->start_offset(), call.callee->end_offset());
      if (!name)
        return reject("LDS helper requires a unique sized local function symbol");
      helper.caller = block;
      helper.callee = call.callee;
      helper.continuation = call.continuation;
      // The exact symbol includes the context, template argument, pointer and site_id signature.
      if (*name == "_ZN7hip_moi7context11lds_load_atIjEET_PKS2_jNS_7site_idE")
        helper.api = AccessKind::Read;
      else if (*name == "_ZN7hip_moi7context12lds_store_atIjEEvPT_S2_jNS_7site_idE")
        helper.api = AccessKind::Write;

      for (const Instruction &instruction : call.callee->instructions()) {
        if (instruction.is_barrier())
          return reject("LDS helper requires a straight-line body without synchronization");
        const InstDefUse effects(instruction, wave_size);
        if (effects.defs.intersects({RegClass::SGPR, call.return_sreg, 2}))
          return reject("LDS helper clobbers its saved return PC");
        // A later memory operation, including one in the opposite direction or segment,
        // supersedes the previous candidate for the API's terminal dereference.
        if (instruction.amdgpu_memory_kind() != amdgpu::MemoryKind::None)
          helper.terminal_access = &instruction;
      }
      // This source-level specialization dereferences one unsigned dword. A symbol
      // alone does not transfer the premise to a different width or addressing form.
      if (helper.terminal_access) {
        const auto op = helper.terminal_access->amdgpu_operation();
        using Op = amdgpu::Operation;
        if (op != Op::FLAT_LOAD_DWORD && op != Op::FLAT_LOAD_B32 && op != Op::FLAT_STORE_DWORD &&
            op != Op::FLAT_STORE_B32)
          helper.terminal_access = nullptr;
      }
    }
  }
  return helper;
}
} // namespace rocjitsu::consan
