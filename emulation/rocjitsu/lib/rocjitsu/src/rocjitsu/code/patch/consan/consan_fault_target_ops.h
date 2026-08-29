// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_target_ops.h
/// @brief Target-normalized fault-mutation capabilities.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu {

enum class ConSanAtomicFaultEncoding : uint8_t {
  Unsupported,
  CdnaFlat,
  FlatLike,
  Buffer,
  Ds,
};

struct ConSanOrdinaryGlobalFaultEncoding {
  int32_t byte_offset = 0;
  uint32_t scope = 0;
};

[[nodiscard]] bool consan_lds_address_fault_arch_supported(rj_code_arch_t arch);

[[nodiscard]] ConSanAtomicFaultEncoding
classify_consan_atomic_fault_encoding(std::string_view mnemonic, uint32_t size,
                                      rj_code_arch_t arch);
[[nodiscard]] bool consan_atomic_fault_supports_scope(ConSanAtomicFaultEncoding encoding);
[[nodiscard]] bool consan_atomic_fault_supports_order(ConSanAtomicFaultEncoding encoding);
[[nodiscard]] bool consan_atomic_fault_supports_address(ConSanAtomicFaultEncoding encoding);

[[nodiscard]] std::optional<ConSanOrdinaryGlobalFaultEncoding>
decode_consan_ordinary_global_fault_encoding(std::span<const uint8_t> instruction);
[[nodiscard]] bool rewrite_consan_ordinary_global_fault_offset(std::span<uint8_t> instruction,
                                                               int32_t byte_offset);
[[nodiscard]] bool rewrite_consan_ordinary_global_fault_scope(std::span<uint8_t> instruction,
                                                              uint32_t scope);

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
