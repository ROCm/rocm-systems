// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_sequence.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <utility>

namespace rocjitsu {

InstructionSequence::Label InstructionSequence::make_label() {
  labels_.push_back(std::nullopt);
  return labels_.size() - 1u;
}

InstructionSequence::Label InstructionSequence::mark_label() {
  const Label label = make_label();
  labels_[label] = words_.size();
  return label;
}

bool InstructionSequence::bind(Label label) {
  if (label >= labels_.size() || labels_[label])
    return false;
  labels_[label] = words_.size();
  return true;
}

bool InstructionSequence::emit_branch(Label label, BranchKind kind) {
  if (label >= labels_.size())
    return false;
  branch_fixups_.push_back({words_.size(), label, kind});
  words_.push_back(0u);
  return true;
}

bool InstructionSequence::resolve_branches(rj_code_arch_t arch) {
  std::vector<std::pair<size_t, uint32_t>> resolved;
  resolved.reserve(branch_fixups_.size());
  for (const BranchFixup &fixup : branch_fixups_) {
    if (fixup.label >= labels_.size() || !labels_[fixup.label])
      return false;
    const auto displacement = compute_sopp_branch_simm16(fixup.word_index * sizeof(uint32_t),
                                                         *labels_[fixup.label] * sizeof(uint32_t));
    if (!displacement)
      return false;
    uint32_t word = 0;
    switch (fixup.kind) {
    case BranchKind::Unconditional:
      word = build_s_branch(*displacement, arch);
      break;
    case BranchKind::SccZero: {
      const auto branch = instrumentation::build_s_cbranch_scc0(*displacement, arch);
      if (!branch)
        return false;
      word = *branch;
      break;
    }
    case BranchKind::SccNonzero: {
      const auto branch = instrumentation::build_s_cbranch_scc1(*displacement, arch);
      if (!branch)
        return false;
      word = *branch;
      break;
    }
    case BranchKind::VccZero: {
      const auto branch = instrumentation::build_s_cbranch_vccz(*displacement, arch);
      if (!branch)
        return false;
      word = *branch;
      break;
    }
    case BranchKind::VccNonzero: {
      const auto branch = instrumentation::build_s_cbranch_vccnz(*displacement, arch);
      if (!branch)
        return false;
      word = *branch;
      break;
    }
    }
    resolved.emplace_back(fixup.word_index, word);
  }
  for (const auto &[word_index, word] : resolved)
    words_[word_index] = word;
  branch_fixups_.clear();
  return true;
}

} // namespace rocjitsu
