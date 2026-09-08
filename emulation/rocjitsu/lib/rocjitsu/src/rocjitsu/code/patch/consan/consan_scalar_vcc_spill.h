// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_scalar_vcc_spill.h
/// @brief Shared transform artifact for preserving borrowed scalar VCC state.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// Representation used to preserve a borrowed scalar VCC-save range in
/// VGPRs. The numeric value is the exact consecutive reservoir width.
enum class ConSanScalarVccReservoir : uint16_t {
  PackedLanes = 1,
  PrivateSpill = 2,
  DynamicStackBootstrap = 4,
};

/// Complete ABI effect of borrowing live scalar registers for VCC.
/// Absence means no borrowed scalar state. This common artifact lets pipeline,
/// diagnostics, and validation carry the effect without depending on the mode
/// that selected it.
struct ConSanScalarVccSpill {
  uint16_t vcc_save_sgpr;
  uint16_t reservoir_vgpr;
  ConSanScalarVccReservoir reservoir;

  [[nodiscard]] uint16_t reservoir_vgpr_count() const { return static_cast<uint16_t>(reservoir); }

  [[nodiscard]] bool is_well_formed() const {
    const uint16_t count = reservoir_vgpr_count();
    return (count == 1u || count == 2u || count == 4u) &&
           static_cast<uint32_t>(reservoir_vgpr) + count <= 256u;
  }
};

} // namespace rocjitsu
