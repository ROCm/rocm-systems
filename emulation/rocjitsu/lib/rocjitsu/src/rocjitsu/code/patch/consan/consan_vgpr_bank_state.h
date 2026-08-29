// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_vgpr_bank_state.h
/// @brief Target-operation interface for selectable VGPR-bank state.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu {

/// Return the active selectable-VGPR-bank mode immediately before one site.
///
/// Targets without selectable VGPR banks and malformed byte ranges return no
/// value. The raw member instruction recipe remains private to its target
/// implementation.
[[nodiscard]] std::optional<uint16_t>
consan_selectable_vgpr_bank_mode_at(rj_code_arch_t arch, std::span<const uint8_t> bytes,
                                    uint64_t text_file_offset, uint64_t container_entry_text_offset,
                                    uint64_t site_file_offset);

/// Return whether an exact byte range contains a selectable-bank transition.
/// Unsupported targets and malformed ranges have no such transition.
[[nodiscard]] bool consan_selectable_vgpr_bank_transition_in_range(rj_code_arch_t arch,
                                                                   std::span<const uint8_t> bytes,
                                                                   uint64_t begin_file_offset,
                                                                   uint64_t end_file_offset);

} // namespace rocjitsu
