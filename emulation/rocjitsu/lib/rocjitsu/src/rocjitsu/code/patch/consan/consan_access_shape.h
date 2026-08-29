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

namespace rocjitsu {

/// Semantic direction of an access decoded from an LDS or FLAT instruction.
///
/// This classification describes what the original instruction does. It does
/// not imply that a particular ConSan engine supports instrumenting the
/// instruction. `Other` preserves decoded instructions that are not ordinary
/// reads, writes, or read-modify-write atomics without misclassifying them.
enum class ConSanLdsAccessKind : uint8_t {
  Read,
  Write,
  Atomic,
  Other,
};

/// Target operation that may consume one normalized access form.
///
/// These names describe mechanisms rather than ConSan engines. All MOI
/// engines replay the guest access, while SuperCollider compares the value
/// observed by a redundant access. Keeping those operations distinct lets one
/// target classifier publish both exact contracts without introducing an
/// engine-by-target matrix.
enum class ConSanAccessLoweringOperation : uint8_t {
  ReplayGuestAccess,
  CompareObservedValue,
  Count,
};

/// Architecture-normalized encoding family for one decoded access.
enum class ConSanAccessLoweringFormKind : uint8_t {
  NativeSingleRange,
  NativeTwoRange,
  FlatVectorAddress,
  FlatScalarVectorAddress,
  DirectToLdsExplicitAddress,
  DirectToLdsLaneAddressed,
  Count,
};

/// Typed reason why the target classifier could not provide an operation.
///
/// Semantic relevance, provenance policy, resource pressure, and placement do
/// not belong here. This enum is solely about exact decoded form and operand
/// lowerability.
enum class ConSanAccessClassifierReason : uint8_t {
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
struct ConSanAccessLoweringForm {
  ConSanAccessLoweringFormKind kind = ConSanAccessLoweringFormKind::Count;
  ConSanLdsAccessKind access_kind = ConSanLdsAccessKind::Other;
  uint32_t instruction_size = 0;
  uint32_t element_width_bits = 0;
  uint32_t range_count = 0;
  uint32_t encoded_offset_scale_bytes = 0;
  uint16_t element_register_count = 0;
  uint16_t data_register_count = 0;
  uint16_t destination_register_count = 0;
  uint16_t address_vgpr_count = 0;
  std::optional<uint16_t> address_vgpr;
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> destination_accvgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::optional<uint16_t> scalar_address_sgpr;
  std::optional<int32_t> immediate_byte_offset;
  bool scale_immediate = false;

  bool operator==(const ConSanAccessLoweringForm &) const = default;
};

/// Exact classifier result for one target access operation.
struct ConSanAccessOperationSupport {
  ConSanAccessClassifierReason reason = ConSanAccessClassifierReason::TargetUnavailable;

  [[nodiscard]] bool available() const { return reason == ConSanAccessClassifierReason::None; }

  bool operator==(const ConSanAccessOperationSupport &) const = default;
};

/// One authoritative classification of a decoded access for every current
/// access-lowering mechanism.
struct ConSanAccessLoweringClassification {
  std::optional<ConSanAccessLoweringForm> form;
  ConSanAccessClassifierReason normalization_reason =
      ConSanAccessClassifierReason::TargetUnavailable;
  ConSanAccessOperationSupport replay_guest_access;
  ConSanAccessOperationSupport compare_observed_value;

  [[nodiscard]] const ConSanAccessOperationSupport &
  operation(ConSanAccessLoweringOperation value) const {
    return value == ConSanAccessLoweringOperation::CompareObservedValue ? compare_observed_value
                                                                        : replay_guest_access;
  }

  [[nodiscard]] bool normalized() const {
    return form.has_value() && normalization_reason == ConSanAccessClassifierReason::None;
  }

  bool operator==(const ConSanAccessLoweringClassification &) const = default;
};

namespace consan_detail {

/// Decoder-owned static shape of one native LDS instruction carrying two
/// addresses.
///
/// Each encoded offset selects an independent range with
/// `element_width_bits` bits. `offset_scale_bytes` converts either eight-bit
/// encoded offset to a byte displacement. `kind` identifies whether the two
/// ranges are read or written. This record only describes decoded instruction
/// geometry; it does not admit the instruction for any ConSan mechanism.
struct DecodedNativeLdsTwoRangeShape {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
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
      NamedForm{"ds_load_2addr_b32", {ConSanLdsAccessKind::Read, 32u, 4u}},
      NamedForm{"ds_store_2addr_b32", {ConSanLdsAccessKind::Write, 32u, 4u}},
      NamedForm{"ds_read2_b32", {ConSanLdsAccessKind::Read, 32u, 4u}},
      NamedForm{"ds_write2_b32", {ConSanLdsAccessKind::Write, 32u, 4u}},
      NamedForm{"ds_load_2addr_b64", {ConSanLdsAccessKind::Read, 64u, 8u}},
      NamedForm{"ds_store_2addr_b64", {ConSanLdsAccessKind::Write, 64u, 8u}},
      NamedForm{"ds_read2_b64", {ConSanLdsAccessKind::Read, 64u, 8u}},
      NamedForm{"ds_write2_b64", {ConSanLdsAccessKind::Write, 64u, 8u}},
      NamedForm{"ds_load_2addr_stride64_b32", {ConSanLdsAccessKind::Read, 32u, 256u}},
      NamedForm{"ds_store_2addr_stride64_b32", {ConSanLdsAccessKind::Write, 32u, 256u}},
      NamedForm{"ds_read2st64_b32", {ConSanLdsAccessKind::Read, 32u, 256u}},
      NamedForm{"ds_write2st64_b32", {ConSanLdsAccessKind::Write, 32u, 256u}},
      NamedForm{"ds_load_2addr_stride64_b64", {ConSanLdsAccessKind::Read, 64u, 512u}},
      NamedForm{"ds_store_2addr_stride64_b64", {ConSanLdsAccessKind::Write, 64u, 512u}},
      NamedForm{"ds_read2st64_b64", {ConSanLdsAccessKind::Read, 64u, 512u}},
      NamedForm{"ds_write2st64_b64", {ConSanLdsAccessKind::Write, 64u, 512u}},
  };
  const auto form = std::ranges::find(forms, mnemonic, &NamedForm::mnemonic);
  return form == forms.end() ? std::nullopt
                             : std::optional<DecodedNativeLdsTwoRangeShape>(form->shape);
}

} // namespace consan_detail
} // namespace rocjitsu
