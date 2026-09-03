// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// How SuperCollider preserves a borrowed scalar VCC-save range in VGPRs.
/// The numeric value is the exact consecutive reservoir width.
enum class ConSanSuperColliderScalarVccReservoir : uint16_t {
  PackedLanes = 1,
  PrivateSpill = 2,
  DynamicStackBootstrap = 4,
};

/// Complete mode-local effect of borrowing live scalar registers for VCC.
/// Absence means no borrowed scalar state. Dynamic-stack bootstrapping also
/// uses the wider reservoir to preserve incoming frame state and SCC.
struct ConSanSuperColliderScalarVccSpill {
  uint16_t vcc_save_sgpr;
  uint16_t reservoir_vgpr;
  ConSanSuperColliderScalarVccReservoir reservoir;

  [[nodiscard]] uint16_t reservoir_vgpr_count() const { return static_cast<uint16_t>(reservoir); }

  [[nodiscard]] bool is_well_formed() const {
    const uint16_t count = reservoir_vgpr_count();
    return (count == 1u || count == 2u || count == 4u) &&
           static_cast<uint32_t>(reservoir_vgpr) + count <= 256u;
  }
};
