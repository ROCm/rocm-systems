// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_access_shape.h
/// @brief Shared architecture-neutral instruction-shape facts for ConSan.

#pragma once

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_flat_access.h"

#include <array>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string_view>

namespace rocjitsu::consan {

/// Semantic direction of an access decoded from an LDS or FLAT instruction.
///
/// This classification describes what the original instruction does. It does
/// not imply that a particular ConSan mode supports instrumenting the
/// instruction. `Other` preserves decoded instructions that are not ordinary
/// reads, writes, or read-modify-write atomics without misclassifying them.
enum class LdsAccessKind : uint8_t {
  Read,
  Write,
  Atomic,
  Other,
};

/// Target operation that may consume one normalized access form.
///
/// These names describe mechanisms rather than ConSan engines. All ConSan
/// engines replay the guest access, while SuperCollider compares the value
/// observed by a redundant access. Keeping those operations distinct lets one
/// target classifier publish both exact contracts without introducing an
/// mode-by-target matrix.
enum class AccessLoweringOperation : uint8_t {
  ReplayGuestAccess,
  CompareObservedValue,
  Count,
};

/// Architecture-normalized encoding family for one decoded access.
enum class AccessLoweringFormKind : uint8_t {
  NativeSingleRange,
  NativeTwoRange,
  FlatVectorAddress,
  FlatScalarVectorAddress,
  DirectToLdsExplicitAddress,
  DirectToLdsLaneAddressed,
  Count,
};

/// Placement of a sub-dword memory value in its source or destination VGPR.
///
/// The classifier owns the instruction spelling that selects this placement.
/// Emitters use the normalized value when they must isolate a compared byte or
/// halfword; `WholeRegister` also covers instructions that extend a narrow
/// memory value to a complete register result.
enum class AccessRegisterValuePlacement : uint8_t {
  WholeRegister,
  Low16,
  High16,
};

/// Typed reason why the target classifier could not provide an operation.
///
/// Semantic relevance, provenance policy, resource pressure, and placement do
/// not belong here. This enum is solely about exact decoded form and operand
/// lowerability.
enum class AccessClassifierReason : uint8_t {
  None,
  NonAccessInstruction,
  InvalidInstructionSize,
  InvalidAccessWidth,
  MissingAddressOperand,
  RangeEncodingUnavailable,
  InstructionOutOfBounds,
  UnsupportedMnemonic,
  UnsupportedEncoding,
  NonzeroImmediateOffset,
  ReservedAddressRegister,
  MissingResultOperand,
  MissingDataOperand,
  OperandRegisterRange,
  TargetUnavailable,
  Count,
};

/// Exact target-normalized form consumed by access lowering mechanisms.
///
/// Raw instruction spellings remain in semantic inventory for diagnostics,
/// but consumers should use this value for range geometry, operand shape, and
/// target-address form instead of classifying the mnemonic again.
struct AccessLoweringForm {
  AccessLoweringFormKind kind = AccessLoweringFormKind::Count;
  LdsAccessKind access_kind = LdsAccessKind::Other;
  uint32_t instruction_size = 0;
  uint32_t element_width_bits = 0;
  uint32_t range_count = 0;
  uint32_t encoded_offset_scale_bytes = 0;
  uint16_t element_register_count = 0;
  uint16_t data_register_count = 0;
  uint16_t destination_register_count = 0;
  uint16_t address_vgpr_count = 0;
  uint16_t data_register_alignment = 1;
  uint16_t destination_allocation_headroom = 0;
  AccessRegisterValuePlacement register_value_placement =
      AccessRegisterValuePlacement::WholeRegister;
  bool destination_preserves_unwritten_bits = false;
  std::optional<uint16_t> address_vgpr;
  std::optional<uint16_t> direct_memory_address_vgpr;
  uint16_t direct_memory_address_vgpr_count = 1;
  uint32_t direct_m0_address_mask = 0xffffffffu;
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> destination_accvgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::optional<uint16_t> scalar_address_sgpr;
  std::optional<int32_t> immediate_byte_offset;
  bool scale_immediate = false;

  bool operator==(const AccessLoweringForm &) const = default;
};

/// Exact classifier result for one target access operation.
struct AccessOperationSupport {
  AccessClassifierReason reason = AccessClassifierReason::TargetUnavailable;

  [[nodiscard]] bool available() const { return reason == AccessClassifierReason::None; }

  bool operator==(const AccessOperationSupport &) const = default;
};

/// One authoritative classification of a decoded access for every current
/// access-lowering mechanism.
struct AccessLoweringClassification {
  std::optional<AccessLoweringForm> form;
  AccessClassifierReason normalization_reason = AccessClassifierReason::TargetUnavailable;
  AccessOperationSupport replay_guest_access;
  AccessOperationSupport compare_observed_value;

  [[nodiscard]] const AccessOperationSupport &operation(AccessLoweringOperation value) const {
    return value == AccessLoweringOperation::CompareObservedValue ? compare_observed_value
                                                                  : replay_guest_access;
  }

  [[nodiscard]] bool normalized() const {
    return form.has_value() && normalization_reason == AccessClassifierReason::None;
  }

  bool operator==(const AccessLoweringClassification &) const = default;
};

namespace detail {

/// Decoder-owned static shape of one native LDS instruction carrying two
/// addresses.
///
/// Each encoded offset selects an independent range with
/// `element_width_bits` bits. `offset_scale_bytes` converts either eight-bit
/// encoded offset to a byte displacement. `kind` identifies whether the two
/// ranges are read or written. This record only describes decoded instruction
/// geometry; it does not admit the instruction for any ConSan mechanism.
struct DecodedNativeLdsTwoRangeShape {
  LdsAccessKind kind = LdsAccessKind::Other;
  uint32_t element_width_bits = 0;
  uint32_t offset_scale_bytes = 0;

  bool operator==(const DecodedNativeLdsTwoRangeShape &) const = default;
};

/// Decode the shared geometry of a two-address native LDS mnemonic.
[[nodiscard]] inline std::optional<DecodedNativeLdsTwoRangeShape>
decode_native_lds_two_range_shape(std::string_view mnemonic) {
  struct NamedForm {
    std::string_view mnemonic;
    DecodedNativeLdsTwoRangeShape shape;
  };
  constexpr std::array forms = {
      NamedForm{"ds_load_2addr_b32", {LdsAccessKind::Read, 32u, 4u}},
      NamedForm{"ds_store_2addr_b32", {LdsAccessKind::Write, 32u, 4u}},
      NamedForm{"ds_read2_b32", {LdsAccessKind::Read, 32u, 4u}},
      NamedForm{"ds_write2_b32", {LdsAccessKind::Write, 32u, 4u}},
      NamedForm{"ds_load_2addr_b64", {LdsAccessKind::Read, 64u, 8u}},
      NamedForm{"ds_store_2addr_b64", {LdsAccessKind::Write, 64u, 8u}},
      NamedForm{"ds_read2_b64", {LdsAccessKind::Read, 64u, 8u}},
      NamedForm{"ds_write2_b64", {LdsAccessKind::Write, 64u, 8u}},
      NamedForm{"ds_load_2addr_stride64_b32", {LdsAccessKind::Read, 32u, 256u}},
      NamedForm{"ds_store_2addr_stride64_b32", {LdsAccessKind::Write, 32u, 256u}},
      NamedForm{"ds_read2st64_b32", {LdsAccessKind::Read, 32u, 256u}},
      NamedForm{"ds_write2st64_b32", {LdsAccessKind::Write, 32u, 256u}},
      NamedForm{"ds_load_2addr_stride64_b64", {LdsAccessKind::Read, 64u, 512u}},
      NamedForm{"ds_store_2addr_stride64_b64", {LdsAccessKind::Write, 64u, 512u}},
      NamedForm{"ds_read2st64_b64", {LdsAccessKind::Read, 64u, 512u}},
      NamedForm{"ds_write2st64_b64", {LdsAccessKind::Write, 64u, 512u}},
  };
  const auto form = std::ranges::find(forms, mnemonic, &NamedForm::mnemonic);
  return form == forms.end() ? std::nullopt
                             : std::optional<DecodedNativeLdsTwoRangeShape>(form->shape);
}

} // namespace detail
} // namespace rocjitsu::consan
