// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Exact scalar state and continuation location for one ordinary
/// SuperCollider indirect instrumentation body.
struct ConSanSuperColliderIndirectBodyRoute {
  ConSanIndirectJumpSgprs jump;
  uint16_t vcc_save_sgpr = 0;
  uint64_t return_offset = 0;

  [[nodiscard]] uint32_t minimum_required_sgpr_count(uint16_t vcc_save_width) const {
    return std::max(jump.minimum_required_sgpr_count(),
                    static_cast<uint32_t>(vcc_save_sgpr) + vcc_save_width);
  }

  [[nodiscard]] bool is_well_formed(uint16_t vcc_save_width) const {
    return vcc_save_width != 0u && jump.is_well_formed() &&
           !jump.overlaps(vcc_save_sgpr, vcc_save_width) &&
           minimum_required_sgpr_count(vcc_save_width) <= 65535u;
  }
};
