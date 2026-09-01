// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_support.h
/// @brief Private SuperCollider guest-replay planning helpers.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {

[[nodiscard]] bool is_instrumentable_group_flat_hint(ConSanFlatAddressSpaceHint hint,
                                                     ConSanFlatProvenanceMode mode);
[[nodiscard]] std::optional<uint16_t>
flat_check_trap_compare_vgpr(const ConSanAccessInventorySite &access);
[[nodiscard]] std::optional<uint16_t>
check_trap_compare_vgpr(const ConSanAccessInventorySite &access, uint16_t chunk_index,
                        rj_code_arch_t arch, uint16_t selectable_vgpr_bank_mode);

} // namespace rocjitsu
