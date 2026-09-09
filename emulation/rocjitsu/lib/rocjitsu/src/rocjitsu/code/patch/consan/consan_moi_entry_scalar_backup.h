// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_entry_scalar_backup.h
/// @brief Entry-local scalar ABI preservation contract.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// Register window copied into one entry-local VGPR while either an
/// owner/epoch or private-state prologue borrows scalar ABI state.
///
/// `vgpr` is the wave-local carrier. `sgpr_base` and `sgpr_count` name the
/// contiguous scalar window transferred through its lanes. The same plan is
/// consumed for save and restore, preventing the two halves of preservation
/// from drifting. Limits are supplied explicitly because they are resolved
/// target facts rather than properties of this target-independent record.
struct ConSanMoiEntryScalarBackup {
  uint16_t vgpr = 0;
  uint16_t sgpr_base = 0;
  uint16_t sgpr_count = 0;

  [[nodiscard]] bool is_well_formed(uint16_t vgpr_limit, uint16_t sgpr_limit) const {
    return vgpr < vgpr_limit && sgpr_count != 0u && sgpr_count <= 64u &&
           static_cast<uint32_t>(sgpr_base) + sgpr_count <= sgpr_limit;
  }

  bool operator==(const ConSanMoiEntryScalarBackup &) const = default;
};

} // namespace rocjitsu
