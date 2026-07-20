// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <type_traits>
#include <vector>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

/// Transactionally appends already-encoded instructions to a word stream.
///
/// Instruction builders commonly return either one encoded word, a range of
/// encoded words, or an optional containing either form. This adapter keeps
/// the error propagation and insertion policy in one place. A failed
/// emit_all() rolls the destination back to its original size.
class InstructionSequence {
public:
  using Label = size_t;

  enum class BranchKind : uint8_t { Unconditional, SccZero, SccNonzero, VccZero, VccNonzero };

  explicit InstructionSequence(std::vector<uint32_t> &words) : words_(words) {}

  [[nodiscard]] bool emit(uint32_t word) {
    words_.push_back(word);
    return true;
  }

  template <std::ranges::input_range Range>
    requires std::same_as<std::remove_cv_t<std::ranges::range_value_t<Range>>, uint32_t>
  [[nodiscard]] bool emit(const Range &range) {
    words_.insert(words_.end(), std::ranges::begin(range), std::ranges::end(range));
    return true;
  }

  template <typename Value> [[nodiscard]] bool emit(const std::optional<Value> &value) {
    return value && emit(*value);
  }

  template <typename... Values> [[nodiscard]] bool emit_all(const Values &...values) {
    const size_t initial_size = words_.size();
    if ((emit(values) && ...))
      return true;
    words_.resize(initial_size);
    return false;
  }

  /// Creates a forward label. Bind it exactly once before resolving branches.
  [[nodiscard]] Label make_label();

  /// Creates a label bound to the current end of the sequence.
  [[nodiscard]] Label mark_label();

  [[nodiscard]] bool bind(Label label);

  /// Emits a placeholder for a branch to label.
  [[nodiscard]] bool emit_branch(Label label, BranchKind kind);

  /// Resolves every pending local branch without partially updating on error.
  [[nodiscard]] bool resolve_branches(rj_code_arch_t arch);

private:
  struct BranchFixup {
    size_t word_index;
    Label label;
    BranchKind kind;
  };

  std::vector<uint32_t> &words_;
  std::vector<std::optional<size_t>> labels_;
  std::vector<BranchFixup> branch_fixups_;
};

} // namespace rocjitsu
