// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan.h
/// @brief Entry point for ConSan DBI code-object patching.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rocjitsu/code/patch/consan/consan_flat_access.h"
#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/code/patch/consan/consan_options.h.inc"

#include "rocjitsu/code/patch/consan/consan_code_object_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_resource_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_fault_sync_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_result.h.inc"

/// Return the active VGPR-bank role mask at an instruction in a gfx1250
/// container. The scan is bounded by the owning container entry.
[[nodiscard]] std::optional<uint16_t>
consan_gfx1250_vgpr_msb_mode_at(std::span<const uint8_t> bytes, uint64_t text_file_offset,
                                uint64_t container_entry_text_offset, uint64_t site_file_offset);

/// Return whether SuperCollider can instrument this decoded FLAT LDS site
/// under the requested provenance policy.
[[nodiscard]] bool consan_supercollider_supports_flat_site(const ConSanFlatSite &site,
                                                           ConSanFlatProvenanceMode mode);

/// Return whether SuperCollider can instrument this decoded native-LDS site
/// on the requested architecture.
[[nodiscard]] bool consan_supercollider_supports_lds_site(const ConSanLdsSite &site,
                                                          rj_code_arch_t arch);

} // namespace rocjitsu
