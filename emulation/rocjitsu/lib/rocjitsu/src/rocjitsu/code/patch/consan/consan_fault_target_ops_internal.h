// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_target_ops_internal.h
/// @brief Family-owned classifiers behind fault target operations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_fault_target_ops.h"

namespace rocjitsu::consan_fault_target_detail {

[[nodiscard]] ConSanAtomicFaultEncoding
classify_gfx9_atomic_fault_encoding(std::string_view mnemonic, uint32_t size);

[[nodiscard]] ConSanAtomicFaultEncoding
classify_gfx12_atomic_fault_encoding(std::string_view mnemonic, uint32_t size);

[[nodiscard]] ConSanAtomicFaultRewriteResult
rewrite_gfx12_atomic_fault_address(std::span<uint8_t> instruction,
                                   ConSanAtomicFaultEncoding encoding, uint32_t width_bits,
                                   uint32_t address_delta);
[[nodiscard]] ConSanAtomicFaultRewriteResult
rewrite_gfx12_atomic_fault_scope_to_wave(std::span<uint8_t> instruction,
                                         ConSanAtomicFaultEncoding encoding);

} // namespace rocjitsu::consan_fault_target_detail
