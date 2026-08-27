// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_cfg.h
/// @brief Canonical control-flow-graph construction inputs for ConSan.

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/consan/consan.h"

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <span>
#include <vector>

namespace rocjitsu::consan_detail {

/// Canonical structural inputs used to recover basic blocks for one ConSan
/// program inventory.
///
/// `leaders` includes every executable container entry plus both ends of any
/// already-applied transform whose continuation may receive control. The
/// narrower `kernel_entries` set lets kernel-scope recovery stop at another
/// kernel rather than claiming its blocks. `code_ranges` bounds decoding to
/// parsed function symbols and already-applied transform bodies. All offset
/// vectors are sorted and unique, and empty code ranges are omitted.
struct ConSanCfgBuildInputs {
  /// Every known entry at which basic-block recovery must begin.
  std::vector<uint64_t> leaders;
  /// Kernel entries that form ownership boundaries during scope recovery.
  std::vector<uint64_t> kernel_entries;
  /// Nonempty executable ranges within which instructions may be decoded.
  std::vector<BasicBlock::CodeRange> code_ranges;
};

/// Derive the one canonical CFG input set shared by ConSan analysis and
/// lowering.
///
/// `preapplied_ranges` describes code introduced by an earlier transactional
/// transform. Its entry and optional continuation become leaders even when
/// its body is empty; only a nonempty body becomes a decoding range. Keeping
/// that rule here prevents composition, ownership, and resource planning from
/// recovering subtly different graphs for the same image.
[[nodiscard]] inline ConSanCfgBuildInputs
build_consan_cfg_inputs(const AmdGpuCodeObject &code_object,
                        std::span<const ConSanKernelInfo> kernels,
                        std::span<const ConSanFunctionInfo> functions,
                        std::span<const ConSanPreappliedCodeRange> preapplied_ranges = {}) {
  ConSanCfgBuildInputs result;
  result.leaders.reserve(kernels.size() + functions.size() + 2u * preapplied_ranges.size());
  result.kernel_entries.reserve(kernels.size());
  for (const ConSanKernelInfo &kernel : kernels) {
    if (!kernel.has_text_range)
      continue;
    result.leaders.push_back(kernel.entry_text_offset);
    result.kernel_entries.push_back(kernel.entry_text_offset);
  }
  for (const ConSanFunctionInfo &function : functions)
    result.leaders.push_back(function.entry_text_offset);
  for (const ConSanPreappliedCodeRange &range : preapplied_ranges) {
    result.leaders.push_back(range.text_offset);
    if (range.continuation_text_offset)
      result.leaders.push_back(*range.continuation_text_offset);
  }
  std::ranges::sort(result.leaders);
  result.leaders.erase(std::ranges::unique(result.leaders).begin(), result.leaders.end());
  std::ranges::sort(result.kernel_entries);
  result.kernel_entries.erase(std::ranges::unique(result.kernel_entries).begin(),
                              result.kernel_entries.end());

  result.code_ranges.reserve(code_object.functions().size() + preapplied_ranges.size());
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0u) {
      result.code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
  }
  for (const ConSanPreappliedCodeRange &range : preapplied_ranges) {
    if (range.size != 0u)
      result.code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
  }
  return result;
}

} // namespace rocjitsu::consan_detail
