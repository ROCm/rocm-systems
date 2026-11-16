// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region.h
/// @brief A prepared region pair, independent of the artifacts it was cut from.
///
/// @details This is what a case file stores and what replay rebuilds from: the original
/// machine words and PCs of each side plus the contract. Replay never needs the
/// artifacts or LLVM.
#pragma once

#include "contract.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct RegionInst {
  uint64_t pc = 0;
  std::vector<uint32_t> words;
  std::string text; ///< Disassembly at analysis time (informational).
  /// A comment-only inline-asm block (no words): its index in the function's sequence of
  /// blocks, which pairs it with the other side's block, and its text, which defines it.
  int inline_asm = -1;
  std::string asm_text;
};

enum class RegionStatus { Ready, Unsupported, Inconclusive };
[[nodiscard]] const char *region_status_name(RegionStatus s);

struct RegionSide {
  size_t begin = 0; ///< Index into the kernel's insts() (analysis only).
  size_t end = 0;
  size_t cut = 0; ///< First matched instruction after the region.
  std::vector<RegionInst> insts;
};

struct RegionPair {
  std::string id;
  RegionSide old_side;
  RegionSide new_side;
  std::vector<size_t> hunks; ///< Diff hunks covered.
  RegionStatus status = RegionStatus::Unsupported;
  std::string reason; ///< Rejection or inconclusive reason.
  RegionContract contract;
  /// Hash binding instructions, PCs, contract and execution mode (the prepared-region key).
  std::string prepared_key;
  bool synthetic = false; ///< Built by mutation (error injection), not by analysing a real pair.
  /// Expansion variant of a base region (handoff 5.2): the base id, and how many common
  /// instructions were added before and after it. Empty parent for base regions.
  std::string parent;
  uint32_t context_before = 0;
  uint32_t context_after = 0;

  [[nodiscard]] Json to_json() const;
  static std::optional<RegionPair> from_json(const Json &j, std::string &error);
};

/// @brief Compute RegionPair::prepared_key from words, PCs, contract and mode.
[[nodiscard]] std::string compute_prepared_key(const RegionPair &region);

} // namespace rocjitsu::isa_diff
