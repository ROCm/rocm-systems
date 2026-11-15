// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file diff.h
/// @brief Align two decoded kernels and classify every difference.
///
/// @details Alignment is a longest common subsequence over instruction keys. A key is
/// the raw encoding words, except that a direct branch's 16-bit displacement is masked
/// so a layout shift does not unmatch it. Every matched direct-branch pair is then
/// re-validated: its two targets must themselves be a matched pair, otherwise the
/// branches are unmatched and fall into a control-flow hunk. A textual difference only
/// proposes candidate change sites; region boundaries are decided by the planner.
#pragma once

#include "kernel_model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

enum class ChangeKind { Instruction, ControlFlow, Layout, Metadata, Unexplained };

[[nodiscard]] const char *change_kind_name(ChangeKind kind);

/// A maximal run of unmatched instructions on either side.
struct Hunk {
  size_t old_begin = 0, old_end = 0; ///< [begin, end) into old insts().
  size_t new_begin = 0, new_end = 0; ///< [begin, end) into new insts().
  ChangeKind kind = ChangeKind::Instruction;
  std::string note;
};

/// A difference outside the instruction stream (descriptor, metadata, padding, layout).
struct ArtifactNote {
  ChangeKind kind = ChangeKind::Metadata;
  std::string what;
  /// True if it can change behaviour that P0 cannot test; it then keeps coverage partial.
  bool semantic = true;
};

struct DiffResult {
  std::vector<int64_t> old_to_new; ///< Matched new index per old instruction, or -1.
  std::vector<int64_t> new_to_old;
  std::vector<Hunk> hunks;
  std::vector<ArtifactNote> notes;
  bool alignment_complete = true; ///< False when the LCS budget forced a coarse hunk.
  bool layout_shift = false;
};

/// @brief Align and classify. @p max_lcs_cells bounds the DP table after prefix/suffix trimming.
[[nodiscard]] DiffResult diff_kernels(const KernelModel &old_model, const KernelModel &new_model,
                                      size_t max_lcs_cells = size_t{64} << 20);

} // namespace rocjitsu::isa_diff
