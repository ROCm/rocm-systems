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

/// Wave32 reservoirs use the architected VCC pair for each jump and preserve
/// its live low half in an endpoint-local ordinary SGPR.
struct ConSanSuperColliderWave32RelayScratch {
  uint16_t entry_vcc_save_sgpr = 0;
  uint16_t return_vcc_save_sgpr = 0;
};

/// Wave64 and mixed-wave reservoirs use endpoint-dead ordinary SGPR tuples.
struct ConSanSuperColliderOrdinaryRelayScratch {
  ConSanIndirectJumpSgprs entry_jump;
  ConSanIndirectJumpSgprs return_jump;
};

/// Target-normalized registers consumed by relay encoding and proof. Optional
/// VCC saves are derived solely for the wave32 strategy and are never retained
/// as an alternative route authority.
struct ConSanSuperColliderRelayEncodingScratch {
  ConSanIndirectJumpSgprs entry_jump;
  ConSanIndirectJumpSgprs return_jump;
  std::optional<uint16_t> entry_vcc_save_sgpr;
  std::optional<uint16_t> return_vcc_save_sgpr;
};

/// Complete mode-local control-flow effect for one indirect relay reservoir.
/// The variant makes the wave32-VCC and ordinary-scalar strategies mutually
/// exclusive rather than encoding them as two optional-field schemas.
struct ConSanSuperColliderRelayReservoirRoute {
  std::variant<ConSanSuperColliderWave32RelayScratch, ConSanSuperColliderOrdinaryRelayScratch>
      scratch;

  [[nodiscard]] bool uses_wave32_vcc() const {
    return std::holds_alternative<ConSanSuperColliderWave32RelayScratch>(scratch);
  }

  [[nodiscard]] ConSanSuperColliderRelayEncodingScratch
  encoding_scratch(ConSanIndirectJumpSgprs wave32_jump) const {
    if (const auto *wave32 = std::get_if<ConSanSuperColliderWave32RelayScratch>(&scratch)) {
      return {wave32_jump, wave32_jump, wave32->entry_vcc_save_sgpr, wave32->return_vcc_save_sgpr};
    }
    const auto &ordinary = std::get<ConSanSuperColliderOrdinaryRelayScratch>(scratch);
    return {ordinary.entry_jump, ordinary.return_jump, std::nullopt, std::nullopt};
  }

  [[nodiscard]] uint32_t minimum_required_sgpr_count() const {
    if (const auto *wave32 = std::get_if<ConSanSuperColliderWave32RelayScratch>(&scratch)) {
      return std::max(static_cast<uint32_t>(wave32->entry_vcc_save_sgpr) + 1u,
                      static_cast<uint32_t>(wave32->return_vcc_save_sgpr) + 1u);
    }
    const auto &ordinary = std::get<ConSanSuperColliderOrdinaryRelayScratch>(scratch);
    return std::max(ordinary.entry_jump.minimum_required_sgpr_count(),
                    ordinary.return_jump.minimum_required_sgpr_count());
  }

  [[nodiscard]] bool is_well_formed() const {
    const auto *ordinary = std::get_if<ConSanSuperColliderOrdinaryRelayScratch>(&scratch);
    return minimum_required_sgpr_count() <= 65535u &&
           (!ordinary ||
            (ordinary->entry_jump.is_well_formed() && ordinary->return_jump.is_well_formed()));
  }
};
