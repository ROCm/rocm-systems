// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_inventory_diagnostics.h
/// @brief Stable diagnostic projections of semantic inventory facts.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <string>

namespace rocjitsu {

[[nodiscard]] std::string consan_fixed_hex(uint64_t value, unsigned digits);
[[nodiscard]] std::string consan_atomic_semantic_role(const ConSanAtomicSite &site);
[[nodiscard]] std::string consan_barrier_decoded_operands(const ConSanBarrierSite &site,
                                                          uint32_t encoding);
[[nodiscard]] std::string consan_atomic_decoded_operands(const ConSanAtomicSite &site);
[[nodiscard]] std::string consan_lds_decoded_operands(const ConSanAccessOperandFacts &operands);
[[nodiscard]] std::string
consan_ordinary_memory_decoded_operands(const ConSanOrdinaryMemorySite &site);

} // namespace rocjitsu
