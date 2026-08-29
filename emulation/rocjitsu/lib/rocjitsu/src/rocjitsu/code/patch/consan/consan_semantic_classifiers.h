// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_semantic_classifiers.h
/// @brief Target-neutral instruction classifiers shared across ConSan stages.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace rocjitsu {

class Instruction;

[[nodiscard]] bool starts_with_any(std::string_view value,
                                   std::initializer_list<std::string_view> prefixes);
[[nodiscard]] bool is_ds_atomic(std::string_view mnemonic);
[[nodiscard]] bool is_ds_read(std::string_view mnemonic);
[[nodiscard]] bool is_ds_write(std::string_view mnemonic);
[[nodiscard]] bool is_ds_register_lane_operation(std::string_view mnemonic);
[[nodiscard]] ConSanLdsAccessKind lds_access_kind(std::string_view mnemonic);
[[nodiscard]] ConSanLdsAccessKind flat_access_kind(std::string_view mnemonic);
[[nodiscard]] uint32_t lds_width_bits(std::string_view mnemonic);
[[nodiscard]] bool is_fence_like(std::string_view mnemonic);
[[nodiscard]] bool is_atomic_instruction(const Instruction &instruction);

/// Exclude instructions whose relocation would carry memory, synchronization,
/// matrix, accumulator, or predicated-definition semantics into a proof cave.
[[nodiscard]] bool has_unsafe_proof_trampoline_flags(const Instruction &instruction);

/// Prefer simple one-word floating-point ALU instructions as proof anchors.
[[nodiscard]] bool is_preferred_proof_anchor(const Instruction &instruction);

} // namespace rocjitsu
