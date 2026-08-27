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

namespace consan_detail {

/// Complete static shape of one native LDS instruction carrying two addresses.
///
/// Each encoded offset selects an independent range with
/// `element_width_bits` bits. `offset_scale_bytes` converts either eight-bit
/// encoded offset to a byte displacement. `kind` identifies whether the two
/// ranges are read or written. Keeping these related facts in one record makes
/// inventory construction, placement, and lowering consume one vocabulary.
struct NativeLdsTwoAddressForm {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  uint32_t element_width_bits = 0;
  uint32_t offset_scale_bytes = 0;

  bool operator==(const NativeLdsTwoAddressForm &) const = default;
};

/// Return the shared shape of a supported two-address native LDS mnemonic.
[[nodiscard]] inline std::optional<NativeLdsTwoAddressForm>
native_lds_two_address_form(std::string_view mnemonic) {
  struct NamedForm {
    std::string_view mnemonic;
    NativeLdsTwoAddressForm form;
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
  return form == forms.end() ? std::nullopt : std::optional<NativeLdsTwoAddressForm>(form->form);
}

/// Return the byte scale between two independently addressed LDS ranges.
[[nodiscard]] inline std::optional<uint32_t>
two_address_native_lds_offset_scale(std::string_view mnemonic) {
  const auto form = native_lds_two_address_form(mnemonic);
  return form ? std::optional<uint32_t>(form->offset_scale_bytes) : std::nullopt;
}

/// Return whether one native LDS mnemonic has the single-range operand shape
/// understood by every MOI access lowerer on `arch`.
///
/// This is both the semantic-policy capability predicate and the lowering
/// admission predicate. Keeping it here prevents a newly supported mnemonic
/// from being planned without an emitter, or emitted without appearing in the
/// typed observation plan.
[[nodiscard]] inline bool is_single_range_native_lds_mnemonic(std::string_view mnemonic,
                                                              rj_code_arch_t arch) {
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       consan_uses_gfx12_encoding(arch)) &&
      (mnemonic == "ds_load_b96" || mnemonic == "ds_store_b96")) {
    return true;
  }
  if (consan_uses_gfx9_cdna_encoding(arch) &&
      (mnemonic == "ds_read_b96" || mnemonic == "ds_write_b96")) {
    return true;
  }
  constexpr std::array always = {
      "ds_load_i8",         "ds_load_u8",          "ds_load_i16",       "ds_load_u16",
      "ds_load_u8_d16",     "ds_load_u8_d16_hi",   "ds_load_i8_d16",    "ds_load_i8_d16_hi",
      "ds_load_u16_d16",    "ds_load_u16_d16_hi",  "ds_store_b8",       "ds_store_b16",
      "ds_store_b8_d16_hi", "ds_store_b16_d16_hi", "ds_read_u8",        "ds_read_u16",
      "ds_write_b8",        "ds_write_b16",        "ds_load_b32",       "ds_load_b64",
      "ds_load_b128",       "ds_load_tr8_b64",     "ds_load_tr16_b128", "ds_read_b32",
      "ds_read_b64",        "ds_read_b64_tr_b16",  "ds_read_b128",      "ds_store_b32",
      "ds_store_b64",       "ds_store_b128",       "ds_write_b32",      "ds_write_b64",
      "ds_write_b128",      "ds_add_f32",          "ds_add_f64",        "ds_add_u32",
      "ds_add_u64",         "ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32",
  };
  if (std::ranges::find(always, mnemonic) != always.end())
    return true;
  constexpr std::array gfx9 = {
      "ds_read_i8",         "ds_read_u8_d16",     "ds_read_u8_d16_hi",
      "ds_read_i8_d16",     "ds_read_i8_d16_hi",  "ds_read_u16_d16",
      "ds_read_u16_d16_hi", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi",
  };
  return consan_uses_gfx9_cdna_encoding(arch) && std::ranges::find(gfx9, mnemonic) != gfx9.end();
}

/// Return whether one decoded FLAT mnemonic has an LDS access shape supported
/// by every MOI engine. Address-space provenance and target operand legality
/// are separate policy checks and deliberately do not enter this vocabulary.
[[nodiscard]] inline bool is_supported_moi_flat_access_mnemonic(std::string_view mnemonic) {
  if (consan_flat_load_subword_semantics(mnemonic) || consan_flat_store_subword_semantics(mnemonic))
    return true;
  constexpr std::array forms = {
      "flat_load_b32",     "flat_load_b64",    "flat_load_b128",     "flat_store_b32",
      "flat_store_b64",    "flat_store_b128",  "flat_load_dword",    "flat_load_dwordx2",
      "flat_load_dwordx4", "flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx4",
      "flat_load_ushort",  "flat_store_short", "flat_load_u16",      "flat_store_b16",
  };
  return std::ranges::find(forms, mnemonic) != forms.end();
}

} // namespace consan_detail
} // namespace rocjitsu
