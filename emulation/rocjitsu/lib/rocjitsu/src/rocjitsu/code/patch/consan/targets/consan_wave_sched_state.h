// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_wave_sched_state.h
/// @brief Target-operation interface for guest wave-scheduling state.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu::consan {

/// Return the WAVE_SCHED_MODE established immediately before one site.
///
/// Targets without a programmable wave-scheduling mode and malformed byte
/// ranges return no value. The instruction encoding remains target-local.
[[nodiscard]] std::optional<uint16_t>
wave_sched_mode_at(rj_code_arch_t arch, std::span<const uint8_t> bytes, uint64_t text_file_offset,
                   uint64_t container_entry_text_offset, uint64_t site_file_offset);

} // namespace rocjitsu::consan
