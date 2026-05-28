// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Coalescer.h
/// @brief Self-contained memory coalescer — turns a per-lane address stream into
/// the distinct set of cache-line transactions it generates.
///
/// Pure + self-included by design: depends only on the rocjitsu-free MemAccess
/// payload (InstrEvent.h) and the standard library. No rocjitsu types, no
/// UarchConfig, no MemorySystem back-reference — so it is unit-testable in complete
/// isolation, and the rocjitsu adapter never needs cache geometry. Re-run per cache
/// level with that level's line_bytes (the reason raw addresses, not pre-coalesced
/// lines, cross the adapter -> lib boundary).

#pragma once

#include "cycle_model/InstrEvent.h"

#include <cstdint>
#include <vector>

namespace cycle_model {

/// Expand each active lane's [addr, addr+elem_bytes) byte range into every
/// `line_bytes`-aligned line it touches (so size-spanning accesses — a 64B s_load,
/// an edge-straddling b128 — count all touched lines), dedup, and return the
/// distinct line base addresses. `line_bytes` must be a power of two.
std::vector<uint64_t> coalesce(const MemAccess& a, uint32_t line_bytes);

}  // namespace cycle_model
