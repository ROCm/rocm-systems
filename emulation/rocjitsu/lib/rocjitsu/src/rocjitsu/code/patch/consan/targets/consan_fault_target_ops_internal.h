// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_target_ops_internal.h
/// @brief Family-owned classifiers behind fault target operations.

#pragma once

#include "rocjitsu/code/patch/consan/targets/consan_fault_target_ops.h"

namespace rocjitsu::consan::fault_target_detail {

[[nodiscard]] AtomicFaultEncoding
classify_cdna3_cdna4_atomic_fault_encoding(std::string_view mnemonic, uint32_t size);

[[nodiscard]] AtomicFaultEncoding
classify_rdna4_cdna5_atomic_fault_encoding(std::string_view mnemonic, uint32_t size);

[[nodiscard]] AtomicFaultRewriteResult
rewrite_rdna4_cdna5_atomic_fault_address(std::span<uint8_t> instruction,
                                         AtomicFaultEncoding encoding, uint32_t width_bits,
                                         uint32_t address_delta);
[[nodiscard]] AtomicFaultRewriteResult
rewrite_rdna4_cdna5_atomic_fault_scope_to_wave(std::span<uint8_t> instruction,
                                               AtomicFaultEncoding encoding);

} // namespace rocjitsu::consan::fault_target_detail
