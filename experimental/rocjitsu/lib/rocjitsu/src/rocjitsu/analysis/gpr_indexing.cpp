// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/gpr_indexing.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <string_view>

namespace rocjitsu {

namespace {

enum class GprIndexingEffect {
  None,
  Enable,
  Disable,
};

[[nodiscard]] GprIndexingEffect gpr_indexing_effect(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_set_gpr_idx_on")
    return GprIndexingEffect::Enable;
  if (mnemonic == "s_set_gpr_idx_off")
    return GprIndexingEffect::Disable;
  return GprIndexingEffect::None;
}

[[nodiscard]] bool transfer_block(BasicBlock &block, bool active) {
  for (const auto &inst : block.instructions()) {
    switch (gpr_indexing_effect(inst)) {
    case GprIndexingEffect::Enable:
      active = true;
      break;
    case GprIndexingEffect::Disable:
      active = false;
      break;
    case GprIndexingEffect::None:
      break;
    }
  }
  return active;
}

} // namespace

GprIndexingAnalysis::GprIndexingAnalysis(std::span<BasicBlock *const> blocks) { analyze(blocks); }

void GprIndexingAnalysis::analyze(std::span<BasicBlock *const> blocks) {
  active_in_.assign(blocks.size(), false);
  active_out_.assign(blocks.size(), false);

  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i], i);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < blocks.size(); ++i) {
      BasicBlock *block = blocks[i];
      if (block == nullptr)
        continue;

      bool active_in = false;
      for (const BasicBlock *pred : block->predecessors()) {
        auto pred_idx = block_index_.find(pred);
        if (pred_idx != block_index_.end())
          active_in = active_in || active_out_[pred_idx->second];
      }

      const bool active_out = transfer_block(*block, active_in);
      if (active_in_[i] != active_in || active_out_[i] != active_out) {
        active_in_[i] = active_in;
        active_out_[i] = active_out;
        changed = true;
      }
    }
  }

  for (size_t i = 0; i < blocks.size(); ++i) {
    BasicBlock *block = blocks[i];
    if (block == nullptr)
      continue;

    bool active = active_in_[i];
    for (const auto &inst : block->instructions()) {
      active_before_.emplace(&inst, active);
      switch (gpr_indexing_effect(inst)) {
      case GprIndexingEffect::Enable:
        active = true;
        break;
      case GprIndexingEffect::Disable:
        active = false;
        break;
      case GprIndexingEffect::None:
        break;
      }
    }
  }
}

bool GprIndexingAnalysis::may_be_active_before(const Instruction &inst) const {
  auto it = active_before_.find(&inst);
  return it != active_before_.end() && it->second;
}

} // namespace rocjitsu
