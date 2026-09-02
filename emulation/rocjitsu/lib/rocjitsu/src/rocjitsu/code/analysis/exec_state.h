// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file exec_state.h
/// @brief Forward "EXEC is provably full" dataflow over one kernel CFG scope.
///
/// @details Liveness must know whether an EXEC-masked vector write overwrites
/// every lane (a real kill) or only the active lanes (inactive lanes keep their
/// old values, so it is not a kill). This pass computes a conservative
/// approximation of the EXEC mask at each program point as a two-point lattice.

#pragma once

#include "rocjitsu/code/analysis/liveness.h" // KernelBlockScope, ScopedCfgEdge

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief Approximated EXEC mask state at a program point.
enum class ExecState : uint8_t {
  Full,
  Unknown,
};

/// @brief Forward "EXEC is provably full" analysis over one kernel CFG scope.
///
/// @details Scope semantics match LivenessAnalysis: only the supplied blocks
/// participate, edges leaving the scope are ignored, and the same scoped
/// call/return `extra_edges` are folded in. Entries are seeded with `Unknown`:
/// the caller-supplied `entry_blocks`, plus any block with no in-scope
/// predecessor.
class ExecMaskAnalysis {
public:
  /// @brief Compute EXEC state for one kernel's block set.
  /// @param wave_size Lanes per wavefront (EXEC bit width): 64 for Wave64, 32
  ///        for Wave32. Used to tell a full-EXEC write from a partial half-write
  ///        (e.g. `s_mov_b32 exec_lo` is the whole mask on Wave32 but only half
  ///        on Wave64). Defaults to 0 and fails if not set.
  /// @param extra_edges Scoped call/return edges folded into the CFG for this
  ///        kernel, matching LivenessAnalysis. EXEC state then flows from a call
  ///        site into the callee and from a return into the continuation, so the
  ///        two analyses agree on which blocks are entries and how EXEC reaches
  ///        them.
  /// @param entry_blocks Real kernel entry blocks, pinned to `Unknown`. A
  ///        hardware entry can be a loop header with an in-scope backedge, so it
  ///        is not recognizable by predecessor count alone; pinning it stops the
  ///        forward meet from deriving `Full` there. Empty falls back to the
  ///        scope leader.
  explicit ExecMaskAnalysis(KernelBlockScope blocks, uint8_t wave_size = 0,
                            std::span<const ScopedCfgEdge> extra_edges = {},
                            std::span<const BasicBlock *const> entry_blocks = {});

  /// @brief EXEC state immediately before @p inst executes.
  /// @returns `ExecState::Unknown` if @p inst was not part of this analysis.
  [[nodiscard]] ExecState before(const Instruction &inst) const;

private:
  void analyze(KernelBlockScope blocks, std::span<const ScopedCfgEdge> extra_edges,
               std::span<const BasicBlock *const> entry_blocks);

  struct BlockExec {
    ExecState in = ExecState::Full;
    ExecState out = ExecState::Full;
    bool is_entry = false;
  };

  uint8_t wave_size_ = 0;
  std::vector<BlockExec> states_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_set<const Instruction *> full_before_;
};

} // namespace rocjitsu
