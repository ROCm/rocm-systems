// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_target_ops.h
/// @brief Target-normalized fault-mutation capabilities.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <string_view>

namespace rocjitsu {

enum class ConSanAtomicFaultEncoding : uint8_t {
  Unsupported,
  CdnaFlat,
  FlatLike,
  Buffer,
  Ds,
};

[[nodiscard]] bool consan_lds_address_fault_arch_supported(rj_code_arch_t arch);

[[nodiscard]] ConSanAtomicFaultEncoding
classify_consan_atomic_fault_encoding(std::string_view mnemonic, uint32_t size,
                                      rj_code_arch_t arch);
[[nodiscard]] bool consan_atomic_fault_supports_scope(ConSanAtomicFaultEncoding encoding);
[[nodiscard]] bool consan_atomic_fault_supports_order(ConSanAtomicFaultEncoding encoding);
[[nodiscard]] bool consan_atomic_fault_supports_address(ConSanAtomicFaultEncoding encoding);

// Concise internal vocabulary retained at existing fault-planning call sites.
using AtomicFaultEncoding = ConSanAtomicFaultEncoding;

[[nodiscard]] inline ConSanAtomicFaultEncoding
atomic_fault_encoding(std::string_view mnemonic, uint32_t size, rj_code_arch_t arch) {
  return classify_consan_atomic_fault_encoding(mnemonic, size, arch);
}

[[nodiscard]] inline bool atomic_fault_supports_scope(ConSanAtomicFaultEncoding encoding) {
  return consan_atomic_fault_supports_scope(encoding);
}

[[nodiscard]] inline bool atomic_fault_supports_order(ConSanAtomicFaultEncoding encoding) {
  return consan_atomic_fault_supports_order(encoding);
}

[[nodiscard]] inline bool atomic_fault_supports_address(ConSanAtomicFaultEncoding encoding) {
  return consan_atomic_fault_supports_address(encoding);
}

} // namespace rocjitsu
