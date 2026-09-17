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

namespace rocjitsu::consan {

enum class AtomicFaultEncoding : uint8_t {
  Unsupported,
  CdnaFlat,
  FlatLike,
  Buffer,
  Ds,
};

struct OrdinaryGlobalFaultEncoding {
  int32_t byte_offset = 0;
  uint32_t scope = 0;
};

/// Target-owned outcome from rewriting one normalized atomic operand.
///
/// The common fault mode renders policy diagnostics from this vocabulary;
/// concrete target packages retain ownership of instruction layout and range
/// checks.
enum class AtomicFaultRewriteStatus : uint8_t {
  Rewritten,
  InvalidEncoding,
  OffsetOverflow,
  MisalignedOffset,
  AlreadyWaveScope,
};

struct AtomicFaultRewriteResult {
  AtomicFaultRewriteStatus status = AtomicFaultRewriteStatus::InvalidEncoding;
  /// Previous target value when a rewrite succeeds. Scope rewriting uses this
  /// to render the semantic before/after diagnostic without exposing raw bits.
  uint32_t previous_value = 0;

  [[nodiscard]] bool rewritten() const { return status == AtomicFaultRewriteStatus::Rewritten; }
};

[[nodiscard]] bool lds_address_fault_arch_supported(rj_code_arch_t arch);

[[nodiscard]] AtomicFaultEncoding
classify_atomic_fault_encoding(std::string_view mnemonic, uint32_t size, rj_code_arch_t arch);
[[nodiscard]] bool atomic_fault_supports_scope(AtomicFaultEncoding encoding);
[[nodiscard]] bool atomic_fault_supports_order(AtomicFaultEncoding encoding);
[[nodiscard]] bool atomic_fault_supports_address(AtomicFaultEncoding encoding);

[[nodiscard]] AtomicFaultRewriteResult rewrite_atomic_fault_address(std::span<uint8_t> instruction,
                                                                    AtomicFaultEncoding encoding,
                                                                    uint32_t width_bits,
                                                                    uint32_t address_delta);
[[nodiscard]] AtomicFaultRewriteResult
rewrite_atomic_fault_scope_to_wave(std::span<uint8_t> instruction, AtomicFaultEncoding encoding);

[[nodiscard]] std::optional<OrdinaryGlobalFaultEncoding>
decode_ordinary_global_fault_encoding(std::span<const uint8_t> instruction);
[[nodiscard]] bool rewrite_ordinary_global_fault_offset(std::span<uint8_t> instruction,
                                                        int32_t byte_offset);
[[nodiscard]] bool rewrite_ordinary_global_fault_scope(std::span<uint8_t> instruction,
                                                       uint32_t scope);

} // namespace rocjitsu::consan
