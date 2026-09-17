// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_inventory_diagnostics.h
/// @brief Stable diagnostic projections of semantic inventory facts.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <string>

namespace rocjitsu::consan {

[[nodiscard]] std::string fixed_hex(uint64_t value, unsigned digits);
[[nodiscard]] std::string atomic_semantic_role(const AtomicSite &site);
[[nodiscard]] std::string barrier_decoded_operands(const BarrierSite &site, uint32_t encoding);
[[nodiscard]] std::string atomic_decoded_operands(const AtomicSite &site);
[[nodiscard]] std::string lds_decoded_operands(const AccessOperandFacts &operands);
[[nodiscard]] std::string ordinary_memory_decoded_operands(const OrdinaryMemorySite &site);

} // namespace rocjitsu::consan
