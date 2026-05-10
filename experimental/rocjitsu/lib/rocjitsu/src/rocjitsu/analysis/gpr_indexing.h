// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpr_indexing.h
/// @brief Forward dataflow for AMDGPU VGPR-indexing mode.
///
/// @details S_SET_GPR_IDX_ON enables a hardware mode where M0 supplies a
/// dynamic offset for selected VALU VGPR operands. While that mode may be
/// active, a printed operand such as v0 is not a complete architectural
/// register reference: the executing instruction may access v0 + M0[7:0].
/// This analysis tracks whether that mode may be active before each decoded
/// instruction. Liveness consumes the result conservatively by protecting the
/// source kernel's allocated ordinary VGPR window from scratch allocation.

#pragma once

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief MAY analysis for whether VGPR indexing is enabled at an instruction.
class GprIndexingAnalysis {
public:
  /// @brief Compute GPR-indexing state over one kernel CFG scope.
  explicit GprIndexingAnalysis(std::span<BasicBlock *const> blocks);

  /// @brief True if VGPR indexing may be active immediately before @p inst.
  [[nodiscard]] bool may_be_active_before(const Instruction &inst) const;

private:
  void analyze(std::span<BasicBlock *const> blocks);

  std::vector<bool> active_in_;
  std::vector<bool> active_out_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, bool> active_before_;
};

} // namespace rocjitsu
