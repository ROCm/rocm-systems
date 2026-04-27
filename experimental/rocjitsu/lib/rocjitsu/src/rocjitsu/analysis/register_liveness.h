// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_liveness.h
/// @brief Per-basic-block VGPR liveness analysis.
///
/// @details Computes which VGPR indices are live at each instruction within a
/// basic block. Used by the semantic translator to verify that register
/// remapping (AccVGPR elimination, MFMA→WMMA expansion) does not clobber
/// live values, and to find free register runs for operand expansion.
///
/// The analysis is intra-block only (no CFG edges or phi nodes). It performs
/// a backward scan from the block's last instruction, building gen/kill sets
/// per instruction. The result maps each instruction offset to the set of
/// VGPR indices live immediately before that instruction executes.

#pragma once

#include <bitset>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace rocjitsu {

class BasicBlock;

/// @brief Maximum VGPR index tracked (covers VGPR 0-255 + AccVGPR 256-511).
inline constexpr uint16_t kMaxVgprIndex = 512;

/// @brief Bitset representing a set of live VGPR indices.
using VgprLiveSet = std::bitset<kMaxVgprIndex>;

/// @brief Per-instruction VGPR liveness information for a basic block.
///
/// @details Constructed via the static `compute()` factory method. Provides
/// queries for liveness at any instruction offset within the block, and
/// utilities for finding free register runs.
class RegisterLiveness {
public:
  /// @brief Compute per-instruction liveness for a basic block.
  ///
  /// @details Walks instructions backward from block end to start. For each
  /// instruction, removes defined (written) VGPR indices from the live set
  /// and adds used (read) VGPR indices. The resulting map gives the live set
  /// immediately before each instruction.
  ///
  /// @param block  The decoded basic block to analyze.
  /// @returns RegisterLiveness with per-instruction live sets.
  [[nodiscard]] static RegisterLiveness compute(BasicBlock &block);

  /// @brief Get the live VGPR set at a given instruction offset.
  /// @param offset  Byte offset of the instruction within .text.
  /// @returns Reference to the live set, or an empty set if offset not found.
  [[nodiscard]] const VgprLiveSet &live_at(uint64_t offset) const;

  /// @brief Check if a specific VGPR index is live at an instruction offset.
  /// @param offset      Byte offset of the instruction within .text.
  /// @param vgpr_index  VGPR index (0-511).
  /// @returns true if the register is live at that point.
  [[nodiscard]] bool is_live(uint64_t offset, uint16_t vgpr_index) const;

  /// @brief Find N consecutive free (not live) VGPR indices.
  ///
  /// @param offset        Byte offset of the instruction within .text.
  /// @param count         Number of consecutive free registers needed.
  /// @param search_start  First VGPR index to consider (default 0).
  /// @returns Base index of the free run, or std::nullopt if none found.
  [[nodiscard]] std::optional<uint16_t> find_free_run(uint64_t offset, uint16_t count,
                                                      uint16_t search_start = 0) const;

  /// @brief Find an even-aligned pair of consecutive free SGPRs.
  /// Uses conservative analysis: allocates above the highest SGPR referenced.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(uint16_t search_start = 0) const;

  /// @brief Find a single free SGPR above all referenced SGPRs.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr(uint16_t search_start = 0) const;

private:
  std::unordered_map<uint64_t, VgprLiveSet> live_sets_;
  VgprLiveSet empty_set_{};
  uint16_t max_sgpr_ = 0; ///< Highest SGPR index referenced in the block.
};

} // namespace rocjitsu
