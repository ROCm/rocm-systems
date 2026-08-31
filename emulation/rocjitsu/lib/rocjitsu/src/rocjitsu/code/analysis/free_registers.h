// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file free_registers.h
/// @brief Free-register search over a RegisterSet of unavailable registers.
///
/// @details This is the allocation half of register selection, split from the
/// analysis half. It answers "where is a run of @p count registers none of which
/// appear in this set", and knows nothing about where the set came from: DBT and
/// DBI semantic lowerings pass a point liveness set from LivenessAnalysis, the
/// whole-kernel usage scan passes its global set, and the DBI trampoline builder
/// passes a set it assembles from callee clobbers and reserved envelope
/// registers.
///
/// The allocation *bound* is deliberately a parameter rather than a constant
/// chosen here.

#pragma once

#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {

/// @brief Lowest base of @p count consecutive registers absent from @p unavailable.
///
/// @param unavailable Registers that must not be selected. Its meaning is the
///        caller's: live-at-a-point, used-anywhere, or reserved-by-the-caller.
/// @param cls Register class to search. Must be one RegisterSet models --
///        SGPR, VGPR, or ACC_VGPR. RegisterSet::contains() answers false for
///        every other class, so searching one would report the whole space free.
/// @param count Number of consecutive registers required; must be non-zero and
///        must fit RegisterRef::width, which the tuple is tested through.
/// @param search_start Lowest candidate base. Rounded up to @p base_alignment.
/// @param base_alignment Required tuple-base alignment; must be a non-zero power
///        of two. Host operands that name a register pair need an even base.
/// @param bound Exclusive upper limit: the whole run must satisfy
///        `base + count <= bound`.
/// @returns The lowest qualifying base, or nullopt when the bound leaves none.
[[nodiscard]] std::optional<uint16_t> find_free_run(const RegisterSet &unavailable, RegClass cls,
                                                    uint16_t count, uint16_t search_start,
                                                    uint16_t base_alignment, uint32_t bound);

/// @brief Lowest single SGPR absent from @p unavailable, below @p bound.
[[nodiscard]] std::optional<uint16_t> find_free_sgpr(const RegisterSet &unavailable, uint32_t bound,
                                                     uint16_t search_start = 0);

/// @brief Lowest even-aligned SGPR pair absent from @p unavailable, below @p bound.
///
/// @details Even alignment is required for pair operations such as saving EXEC
/// with an s_mov_b64-style scalar move.
[[nodiscard]] std::optional<uint16_t>
find_free_sgpr_pair(const RegisterSet &unavailable, uint32_t bound, uint16_t search_start = 0);

} // namespace rocjitsu
