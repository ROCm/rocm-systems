// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file exec_state.h
/// @brief Forward "EXEC is provably full" dataflow over one kernel CFG scope.
///
/// @details Liveness must know whether an EXEC-masked vector write overwrites
/// every lane (a real kill) or only the active lanes (inactive lanes keep their
/// old values, so it is not a kill). This pass computes a conservative
/// approximation of the EXEC mask at each program point as a two-point lattice:
///
///   * `Full`    - EXEC is provably all-ones (every lane active).
///   * `Unknown` - EXEC may be partial.
///
/// It is a *must* analysis: `Full` is only ever reported when it can be proven,
/// so a consumer can safely treat an EXEC-masked write as a kill exactly when
/// the state is `Full`. The lattice meet at a CFG join is `Full` iff every
/// in-scope predecessor is `Full`, and the kernel entry is assumed `Unknown`
/// (a wavefront may launch with a partial mask — e.g. the last wave of a
/// workgroup). Consequently the only thing that introduces `Full` is an
/// instruction that provably writes an all-ones EXEC mask; any other EXEC write
/// (narrowing, restore-from-register, save/restore, v_cmpx, ...) yields
/// `Unknown`.
///
/// The per-instruction EXEC classification is intentionally conservative: an
/// instruction is treated as writing EXEC when it carries the WRITES_EXEC flag
/// or a destination operand resolves to RegClass::EXEC, and as writing an
/// all-ones mask only when its single source is a compile-time all-ones
/// constant. Anything it cannot prove leaves the state `Unknown`, which is the
/// safe direction for the liveness consumer.

#pragma once

#include "rocjitsu/analysis/liveness.h" // KernelBlockScope

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief Approximated EXEC mask state at a program point.
enum class ExecState : uint8_t {
  Full,    ///< EXEC is provably all-ones (every lane active).
  Unknown, ///< EXEC may be partial; the conservative top of the lattice.
};

/// @brief Forward "EXEC is provably full" analysis over one kernel CFG scope.
///
/// @details Scope semantics match LivenessAnalysis: only the supplied blocks
/// participate, and edges leaving the scope are ignored. Blocks with no
/// in-scope predecessor are treated as entries and seeded with `Unknown`.
class ExecMaskAnalysis {
public:
  /// @brief Compute EXEC state for one kernel's block set.
  explicit ExecMaskAnalysis(KernelBlockScope blocks);

  /// @brief EXEC state immediately before @p inst executes.
  /// @returns `ExecState::Unknown` if @p inst was not part of this analysis.
  [[nodiscard]] ExecState before(const Instruction &inst) const;

private:
  void analyze(KernelBlockScope blocks);

  struct BlockExec {
    ExecState in = ExecState::Full;  ///< State entering the block.
    ExecState out = ExecState::Full; ///< State leaving the block.
    bool is_entry = false;           ///< No in-scope predecessor.
  };

  std::vector<BlockExec> states_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, ExecState> before_;
};

} // namespace rocjitsu
