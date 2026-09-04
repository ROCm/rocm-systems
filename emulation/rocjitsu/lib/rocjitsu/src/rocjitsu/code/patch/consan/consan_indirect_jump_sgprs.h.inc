// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace rocjitsu {

/// Scalar pair and SCC snapshot used while establishing a spill frame or
/// reloading owner-local state.
struct ConSanMoiScalarSpillTemporaries {
  uint16_t frame_base_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  auto operator<=>(const ConSanMoiScalarSpillTemporaries &) const = default;
};

} // namespace rocjitsu
