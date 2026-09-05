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

  enum class BranchKind : uint8_t {
    Unconditional,
    SccZero,
    SccNonzero,
    VccZero,
    VccNonzero,
    ExecZero,
    ExecNonzero,
  };

  explicit InstructionSequence(std::vector<uint32_t> &words)
      : words_(words), initial_size_(words.size()) {}

  ~InstructionSequence() {
    if (failed_)
      words_.resize(initial_size_);
  }

  InstructionSequence(const InstructionSequence &) = delete;
  InstructionSequence &operator=(const InstructionSequence &) = delete;

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

  /// Append one transactional batch and remember failure on the sequence.
  ///
  /// Sticky construction lets a caller describe the complete instruction
  /// program without repeating local boolean propagation. The first failure
  /// rolls the destination back to its construction size; later operations
  /// are ignored and finish() remains false.
  template <typename... Values> InstructionSequence &append(const Values &...values) {
    if (!failed_ && !emit_all(values...))
      fail();
    return *this;
  }

  /// Incorporate a helper that emitted directly into the same destination.
  InstructionSequence &require(bool success) {
    if (!failed_ && !success)
      fail();
    return *this;
  }

  InstructionSequence &branch(Label label, BranchKind kind) {
    return require(failed_ || emit_branch(label, kind));
  }

  InstructionSequence &bind_label(Label label) { return require(failed_ || bind(label)); }

  [[nodiscard]] explicit operator bool() const { return !failed_; }

  /// Complete a branch-free transaction.
  [[nodiscard]] bool finish() const { return !failed_; }

  /// Resolve local branches and complete the transaction atomically.
  [[nodiscard]] bool finish(rj_code_arch_t arch) {
    if (!failed_ && !resolve_branches(arch))
      fail();
    return !failed_;
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

  void fail() {
    words_.resize(initial_size_);
    failed_ = true;
  }

  std::vector<uint32_t> &words_;
  size_t initial_size_ = 0;
  bool failed_ = false;
  std::vector<std::optional<size_t>> labels_;
  std::vector<BranchFixup> branch_fixups_;
};

} // namespace rocjitsu
