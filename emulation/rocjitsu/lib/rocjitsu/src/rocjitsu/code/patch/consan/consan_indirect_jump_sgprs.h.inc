// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace rocjitsu {

/// Complete scalar scratch for one SCC-preserving indirect jump.
struct ConSanIndirectJumpSgprs {
  uint16_t pc_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  [[nodiscard]] uint32_t minimum_required_sgpr_count() const {
    return std::max(static_cast<uint32_t>(pc_sgpr) + 2u, static_cast<uint32_t>(scc_save_sgpr) + 1u);
  }

  [[nodiscard]] bool overlaps(uint16_t base, uint16_t width) const {
    const uint32_t end = static_cast<uint32_t>(base) + width;
    return width != 0u && ((pc_sgpr < end && base < static_cast<uint32_t>(pc_sgpr) + 2u) ||
                           (scc_save_sgpr < end && base <= scc_save_sgpr));
  }

  [[nodiscard]] bool is_well_formed() const {
    return minimum_required_sgpr_count() <= 65535u && scc_save_sgpr != pc_sgpr &&
           static_cast<uint32_t>(scc_save_sgpr) != static_cast<uint32_t>(pc_sgpr) + 1u;
  }

  auto operator<=>(const ConSanIndirectJumpSgprs &) const = default;
};

} // namespace rocjitsu
