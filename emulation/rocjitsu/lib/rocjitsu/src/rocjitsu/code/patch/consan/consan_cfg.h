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
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::consan::detail {

/// Canonical structural inputs used to recover basic blocks for one ConSan
/// program inventory.
///
/// `leaders` includes every executable container entry plus both ends of any
/// already-applied transform whose continuation may receive control. The
/// narrower `kernel_entries` set lets kernel-scope recovery stop at another
/// kernel rather than claiming its blocks. `code_ranges` bounds decoding to
/// parsed function symbols and already-applied transform bodies. All offset
/// vectors are sorted and unique, and empty code ranges are omitted.
struct CfgBuildInputs {
  /// Every known entry at which basic-block recovery must begin.
  std::vector<uint64_t> leaders;
  /// Kernel entries that form ownership boundaries during scope recovery.
  std::vector<uint64_t> kernel_entries;
  /// Nonempty executable ranges within which instructions may be decoded.
  std::vector<BasicBlock::CodeRange> code_ranges;
};

/// One inventory call's CFG, bound to immutable image bytes and architecture.
/// Reuse requires exactly the same build arguments. This object must not cross
/// a patched-image or retry boundary; it owns no process-wide cache state.
class ProgramAnalysisCfg {
public:
  ProgramAnalysisCfg(const AmdGpuCodeObject &image, Decoder &decoder, rj_code_arch_t arch)
      : image_(image), decoder_(decoder), arch_(arch) {}

  /// Returned blocks are read-only and remain valid until the next get().
  /// A changed key rebuilds instead of weakening entry/range constraints.
  [[nodiscard]] const std::vector<std::unique_ptr<BasicBlock>> &
  get(std::span<const uint64_t> leaders, std::span<const BasicBlock::CodeRange> ranges) {
    const bool same_ranges =
        std::ranges::equal(ranges_, ranges, [](const auto &lhs, const auto &rhs) {
          return lhs.start_offset == rhs.start_offset && lhs.size == rhs.size;
        });
    if (ready_ && std::ranges::equal(leaders_, leaders) && same_ranges)
      return blocks_;
    // Drop the old graph before constructing its replacement. Failed builds
    // are never reusable, and retaining two whole-library graphs is unnecessary.
    ready_ = false;
    blocks_.clear();
    auto blocks = BasicBlock::build(image_, decoder_, arch_, leaders, ranges);
    leaders_.assign(leaders.begin(), leaders.end());
    ranges_.assign(ranges.begin(), ranges.end());
    blocks_ = std::move(blocks);
    ready_ = true;
    return blocks_;
  }

private:
  const AmdGpuCodeObject &image_;
  Decoder &decoder_;
  rj_code_arch_t arch_;
  bool ready_ = false;
  std::vector<uint64_t> leaders_;
  std::vector<BasicBlock::CodeRange> ranges_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
};

/// Derive the one canonical CFG input set shared by ConSan analysis and
/// lowering.
///
/// `preapplied_ranges` describes code introduced by an earlier transactional
/// transform. Its entry and optional continuation become leaders even when
/// its body is empty; only a nonempty body becomes a decoding range. Keeping
/// that rule here prevents composition, ownership, and resource planning from
/// recovering subtly different graphs for the same image.
[[nodiscard]] inline CfgBuildInputs
build_cfg_inputs(const AmdGpuCodeObject &code_object, std::span<const ProgramContainer> containers,
                 std::span<const PreappliedCodeRange> preapplied_ranges = {}) {
  CfgBuildInputs result;
  result.leaders.reserve(containers.size() + 2u * preapplied_ranges.size());
  result.kernel_entries.reserve(containers.size());
  for (const ProgramContainer &container : containers) {
    if (container.is_kernel() && !container.has_text_range)
      continue;
    result.leaders.push_back(container.entry_text_offset);
    if (container.is_kernel())
      result.kernel_entries.push_back(container.entry_text_offset);
  }
  for (const PreappliedCodeRange &range : preapplied_ranges) {
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
  for (const PreappliedCodeRange &range : preapplied_ranges) {
    if (range.size != 0u)
      result.code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
  }
  return result;
}

/// Derive CFG inputs scoped to selected kernel ranges when ownership proves
/// that no separately symbolized helper can be shared with another kernel.
///
/// The complete kernel-entry set remains in `kernel_entries`, preserving scope
/// boundaries. When function containers exist, or no selection was requested,
/// this returns the ordinary complete input set: ownership through a shared
/// helper must be established before any sibling kernel can be discarded.
[[nodiscard]] inline CfgBuildInputs
build_cfg_inputs_for_selection(const AmdGpuCodeObject &code_object,
                               std::span<const ProgramContainer> containers,
                               std::span<const PreappliedCodeRange> preapplied_ranges,
                               const Request &request, const DebugOverrides &debug) {
  CfgBuildInputs result = build_cfg_inputs(code_object, containers, preapplied_ranges);
  const bool has_selection =
      !request.kernel_name_allowlist.empty() || !debug.test_kernel_name_filter.empty();
  const bool has_function_container =
      std::ranges::any_of(containers, [](const auto &container) { return !container.is_kernel(); });
  if (!has_selection || has_function_container)
    return result;

  result.leaders.clear();
  result.code_ranges.clear();
  result.leaders.reserve(containers.size() + 2u * preapplied_ranges.size());
  result.code_ranges.reserve(containers.size() + preapplied_ranges.size());
  for (const ProgramContainer &kernel : containers) {
    if (!kernel.is_kernel() || !kernel.has_text_range ||
        !container_selected(request, debug, kernel.name)) {
      continue;
    }
    result.leaders.push_back(kernel.entry_text_offset);
    if (kernel.code_size != 0u)
      result.code_ranges.push_back(
          {.start_offset = kernel.entry_text_offset, .size = kernel.code_size});
  }
  for (const PreappliedCodeRange &range : preapplied_ranges) {
    result.leaders.push_back(range.text_offset);
    if (range.continuation_text_offset)
      result.leaders.push_back(*range.continuation_text_offset);
    if (range.size != 0u)
      result.code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
  }
  std::ranges::sort(result.leaders);
  result.leaders.erase(std::ranges::unique(result.leaders).begin(), result.leaders.end());
  return result;
}

} // namespace rocjitsu::consan::detail
