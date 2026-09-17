// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_atomic_classifier.h
/// @brief Target normalization for ordered ConSan atomic operations.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::consan {

struct AtomicSite;

/// Architecture-normalized address family for one ordered atomic operation.
enum class AtomicLoweringFormKind : uint8_t {
  FlatVectorAddress,
  FlatScalarVectorAddress,
  GlobalVectorAddress,
  GlobalScalarVectorAddress,
  LdsVectorOffset,
  BufferResourceVectorOffset,
  Count,
};

/// Exact normalized facts consumed by ordered-atomic target mechanisms.
struct AtomicLoweringForm {
  AtomicLoweringFormKind kind = AtomicLoweringFormKind::Count;
  uint32_t instruction_size = 0;
  uint32_t value_width_bits = 0;
  uint16_t value_register_count = 0;
  uint16_t data_register_count = 0;
  uint16_t destination_register_count = 0;
  uint16_t address_vgpr = 0;
  uint16_t address_vgpr_count = 0;
  uint16_t data_vgpr = 0;
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> scalar_base_sgpr;
  std::optional<uint16_t> scalar_offset_sgpr;
  int32_t signed_byte_offset = 0;
  bool scale_vector_offset = false;
  bool sign_extend_vector_offset = false;
  bool is_rmw = true;
  bool compare_exchange = false;
  bool returns_old_value = false;

  /// Verify the target-normalized shape without reinterpreting raw encoding.
  [[nodiscard]] constexpr bool is_well_formed() const {
    return static_cast<uint8_t>(kind) < static_cast<uint8_t>(AtomicLoweringFormKind::Count) &&
           instruction_size != 0u && value_width_bits != 0u && value_register_count != 0u &&
           data_register_count != 0u && address_vgpr_count != 0u;
  }

  bool operator==(const AtomicLoweringForm &) const = default;
};

/// Typed reason why an ordered-atomic target mechanism cannot consume a site.
enum class AtomicClassifierReason : uint8_t {
  None,
  UnsupportedAddressSource,
  InvalidAccessWidth,
  UnsupportedEncoding,
  NonzeroImmediateOffset,
  MissingOperands,
  UnsupportedInputWidth,
  UnsupportedOffset,
  ResultAddressAlias,
  CompareExchangeOutcomeUnavailable,
  MissingOrderingMetadata,
  UnsupportedScope,
  TargetUnavailable,
  Count,
};

struct AtomicLoweringClassification {
  std::optional<AtomicLoweringForm> form;
  AtomicClassifierReason normalization_reason = AtomicClassifierReason::TargetUnavailable;
  AtomicClassifierReason address_reason = AtomicClassifierReason::TargetUnavailable;
  AtomicClassifierReason exact_ordering_reason = AtomicClassifierReason::TargetUnavailable;
  AtomicClassifierReason causal_ordering_reason = AtomicClassifierReason::TargetUnavailable;

  [[nodiscard]] bool normalized() const {
    return form.has_value() && normalization_reason == AtomicClassifierReason::None;
  }
  [[nodiscard]] bool address_available() const {
    return normalized() && address_reason == AtomicClassifierReason::None;
  }
  [[nodiscard]] bool exact_ordering_available() const {
    return exact_ordering_reason == AtomicClassifierReason::None;
  }
  [[nodiscard]] bool causal_ordering_available() const {
    return causal_ordering_reason == AtomicClassifierReason::None;
  }

  bool operator==(const AtomicLoweringClassification &) const = default;
};

/// Normalize one operand-rich atomic or ordered ordinary-memory decode and
/// classify its address, causal-ordering, and exact-ordering operations. This
/// is the sole authority for raw target encoding admission shared by policy
/// and native lowerers.
[[nodiscard]] AtomicLoweringClassification
classify_atomic_lowering(const AtomicSite &site, rj_code_arch_t arch, bool is_rmw = true);

} // namespace rocjitsu::consan
