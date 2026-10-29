// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/static_barriers.h"

#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {
// SSRC inline-negative encodings start at 193 (-1), as named by the RDNA4
// and CDNA5 OpSelSsrcBarrierId enums. Family policy admits semantic IDs, not
// encodings: sharing this first selector does not admit CDNA5's IDs on RDNA4.
constexpr uint32_t kFirstInlineNegativeIntegerSelector = 193;
constexpr uint32_t kLastInlineNegativeIntegerSelector = 208;

Result reject(StaticBarrierError &error, StaticBarrierErrorCode code, uint64_t offset,
              std::string message) {
  error = {.code = code, .offset = offset, .message = std::move(message)};
  return Result::failure();
}
StaticBarrierMember make_barrier_member(const Instruction &instruction) {
  StaticBarrierMember result = {.offset = instruction.src_loc(),
                                .bytes = std::vector<uint8_t>(instruction.size())};
  std::memcpy(result.bytes.data(), instruction.raw_encoding(), result.bytes.size());
  return result;
}
} // namespace

std::optional<StaticBarrierPolicy> static_barrier_policy(rj_code_target_id_t target,
                                                         const IsaTargetRegistry &registry) {
  const IsaTargetDescriptor *family = registry.find(target);
  if (!family || !registry.find_gpu_target(target))
    return std::nullopt;
  static constexpr std::array<uint32_t, 2> rdna_waves = {32, 64};
  static constexpr std::array<uint32_t, 1> wave32 = {32};
  static constexpr std::array<uint32_t, 1> wave64 = {64};
  static constexpr std::array<StaticBarrierBuiltin, 1> workgroup = {
      {{-1, StaticBarrierScope::Workgroup}}};
  static constexpr std::array<StaticBarrierBuiltin, 2> workgroup_and_cluster = {
      {{-1, StaticBarrierScope::Workgroup}, {-3, StaticBarrierScope::Cluster}}};
  switch (family->architecture_id) {
  case ROCJITSU_CODE_ARCH_RDNA3:
    return StaticBarrierPolicy{
        .form = StaticBarrierForm::FullSingleton, .wave_sizes = rdna_waves, .builtins = workgroup};
  case ROCJITSU_CODE_ARCH_RDNA4:
    return StaticBarrierPolicy{.form = StaticBarrierForm::BuiltinSplitPair,
                               .wave_sizes = rdna_waves,
                               .builtins = workgroup};
  case ROCJITSU_CODE_ARCH_CDNA5:
    return StaticBarrierPolicy{.form = StaticBarrierForm::BuiltinSplitPair,
                               .wave_sizes = wave32,
                               .builtins = workgroup_and_cluster};
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return StaticBarrierPolicy{
        .form = StaticBarrierForm::FullSingleton, .wave_sizes = wave64, .builtins = workgroup};
  default:
    return std::nullopt;
  }
}

FailureOr<std::vector<StaticBarrier>>
qualify_static_barriers(std::span<BasicBlock *const> blocks, rj_code_target_id_t target,
                        const InstructionBoundaries &boundaries, StaticBarrierError &error) {
  error = {};
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(target);
  if (!policy)
    return reject(error, StaticBarrierErrorCode::UnsupportedForm, 0,
                  "target has no complete-barrier mutation policy");
  std::vector<StaticBarrier> barriers;
  for (BasicBlock *block : blocks) {
    // Pairing must stop at a block boundary, even when the next block is adjacent.
    std::optional<StaticBarrier> pending;
    for (const Instruction &instruction : block->instructions()) {
      const std::string_view name = instruction.mnemonic();
      // BARRIER is a coarse flag: decoded init/join currently omit it. Keep the
      // canonical-name guard so unsupported lifecycle/state forms cannot silently
      // pass as ordinary instructions; exact form checks below decide admission.
      if (name.find("barrier") == std::string_view::npos && !instruction.is_barrier())
        continue;
      const uint64_t offset = instruction.src_loc();
      const InstructionBoundaries::Entry *boundary = boundaries.find(offset);
      if (!boundary || boundary->size != instruction.size())
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "barrier does not match the full-text instruction boundaries");
      if (boundary->clause_member || boundary->clause || instruction.size() != 4)
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "barrier must be a complete instruction outside clauses");
      if (policy->form == StaticBarrierForm::FullSingleton) {
        if (name != "s_barrier")
          return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                        "only full s_barrier is qualified for this family");
        barriers.push_back({.form = StaticBarrierForm::FullSingleton,
                            .block_offset = block->start_offset(),
                            .barrier_id = -1,
                            .scope = StaticBarrierScope::Workgroup,
                            .members = {make_barrier_member(instruction)}});
        continue;
      }
      if (name != "s_barrier_signal" && name != "s_barrier_wait")
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "only builtin immediate signal/wait pairs are qualified");
      const uint32_t word = instruction.raw_encoding()[0];
      int32_t id;
      if (name == "s_barrier_signal") {
        const uint32_t selector = word & 0xffu;
        if (selector < kFirstInlineNegativeIntegerSelector ||
            selector > kLastInlineNegativeIntegerSelector)
          return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                        "named, dynamic and literal barrier operands are outside this profile");
        id = -static_cast<int32_t>(selector - kFirstInlineNegativeIntegerSelector + 1);
      } else {
        id = static_cast<int16_t>(word & 0xffffu);
      }
      if (id == -2 || id == -4)
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "trap-only barrier IDs are not user-kernel mutation candidates");
      const auto builtin = std::ranges::find(policy->builtins, id, &StaticBarrierBuiltin::id);
      if (builtin == policy->builtins.end())
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "barrier ID is not a supported user barrier for this family");
      if (name == "s_barrier_signal") {
        if (pending)
          return reject(error, StaticBarrierErrorCode::IncompleteBarrier, offset,
                        "interleaved or repeated signal before completing wait");
        pending = StaticBarrier{.form = StaticBarrierForm::BuiltinSplitPair,
                                .block_offset = block->start_offset(),
                                .barrier_id = id,
                                .scope = builtin->scope,
                                .members = {make_barrier_member(instruction)}};
      } else {
        if (!pending || pending->barrier_id != id)
          return reject(error, StaticBarrierErrorCode::IncompleteBarrier, offset,
                        "wait has no matching same-block signal");
        // Only the admitted signal above creates pending; input mismatches have
        // already returned a diagnostic. Its single member is an internal invariant.
        assert(pending->members.size() == 1);
        pending->members.push_back(make_barrier_member(instruction));
        barriers.push_back(std::move(*pending));
        pending.reset();
      }
    }
    if (pending)
      return reject(error, StaticBarrierErrorCode::IncompleteBarrier,
                    pending->members.front().offset, "signal has no completing same-block wait");
  }
  return barriers;
}
} // namespace rocjitsu
