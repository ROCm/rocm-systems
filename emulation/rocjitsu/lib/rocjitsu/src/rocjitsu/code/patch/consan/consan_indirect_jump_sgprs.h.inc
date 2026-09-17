// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace rocjitsu::consan {

/// Scalar pair and SCC snapshot used while establishing a spill frame or
/// reloading owner-local state.
struct ScalarSpillTemporaries {
  uint16_t frame_base_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  auto operator<=>(const ScalarSpillTemporaries &) const = default;
};

} // namespace rocjitsu::consan
