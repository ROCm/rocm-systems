// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_pregfx12_target_ops.h
/// @brief Shared normalization for the related gfx9-CDNA and gfx11 memory forms.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

template <typename Raw>
ConSanVectorMemoryDecode decode_pregfx12_vector_memory(std::span<const uint8_t> instruction,
                                                       bool global, uint32_t null_saddr) {
  if (instruction.size() < sizeof(Raw))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  Raw raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  const int32_t ioffset = [&] {
    if constexpr (requires { raw.glc; })
      return sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset));
    else
      return global ? sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset))
                    : static_cast<int32_t>(raw.offset);
  }();
  const uint32_t th = [&] {
    if constexpr (requires { raw.glc; })
      return static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u);
    else
      return static_cast<uint32_t>(raw.sc0) | (static_cast<uint32_t>(raw.sc1) << 1u);
  }();
  const bool padding_is_zero = [&] {
    if constexpr (requires { raw.pad_25; })
      return raw.pad_25 == 0u;
    else
      return true;
  }();
  ConSanEncodedFlatSegment segment = ConSanEncodedFlatSegment::Unspecified;
  if (!global && raw.seg == 1u)
    segment = ConSanEncodedFlatSegment::Private;
  else if (!global && raw.seg == 2u)
    segment = ConSanEncodedFlatSegment::Global;
  const bool exact_size = instruction.size() == sizeof(raw);
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding =
          {
              .raw_op = static_cast<uint32_t>(raw.op),
              .raw_saddr = static_cast<uint32_t>(raw.saddr),
              .raw_sve =
                  [&] {
                    if constexpr (requires { raw.sve; })
                      return static_cast<uint32_t>(raw.sve);
                    else
                      return 0u;
                  }(),
              .raw_vaddr = static_cast<uint32_t>(raw.addr),
              .raw_vsrc = static_cast<uint32_t>(raw.data),
              .raw_vdst = static_cast<uint32_t>(raw.vdst),
              .raw_ioffset = ioffset,
              .raw_segment = static_cast<uint32_t>(raw.seg),
              .raw_scope = global ? 2u : 0u,
              .raw_th = th,
              .scope = global ? std::optional{ConSanMemoryScope::Agent} : std::nullopt,
              .encoded_segment = segment,
              .scalar_provenance_sgpr =
                  global && raw.saddr != null_saddr
                      ? std::optional<uint16_t>(static_cast<uint16_t>(raw.saddr))
                      : std::nullopt,
              .scope_follows_address_space = !global,
              .workgroup_acquire_ordering = th == 1u,
              .exact_size = exact_size,
              .ordinary_well_formed = exact_size && raw.encoding == 0x37u &&
                                      raw.seg == (global ? 2u : 0u) && padding_is_zero,
              .ordinary_requires_complete_registers = false,
              .ordinary_mutation_supported = false,
          },
  };
}

template <typename Raw>
void fill_pregfx12_atomic_site(ConSanAtomicSite &site, const Raw &raw, bool global,
                               uint32_t null_saddr) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.addr);
  site.raw_vsrc = static_cast<uint32_t>(raw.data);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  if constexpr (requires { raw.glc; }) {
    site.raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset));
    site.raw_th = static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u);
    site.returns_old_value = raw.glc != 0u;
  } else {
    site.raw_ioffset = global ? sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset))
                              : static_cast<int32_t>(raw.offset);
    site.raw_th = static_cast<uint32_t>(raw.sc0) | (static_cast<uint32_t>(raw.sc1) << 1u);
    site.returns_old_value = raw.sc0 != 0u;
  }
  // Preserve the legacy diagnostic spelling of this implicit device scope.
  site.raw_scope = 2u;
  site.scope = ConSanMemoryScope::Agent;
  if (global) {
    site.saddr_sgpr = raw.saddr == null_saddr
                          ? std::nullopt
                          : std::optional<uint16_t>(static_cast<uint16_t>(raw.saddr));
  }
}

template <typename FlatRaw, typename GlobalRaw>
bool decode_pregfx12_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                 std::span<const uint8_t> instruction, uint32_t null_saddr) {
  if (mnemonic.starts_with("flat_atomic") && instruction.size() >= sizeof(FlatRaw)) {
    FlatRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_pregfx12_atomic_site(site, raw, false, null_saddr);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") && instruction.size() >= sizeof(GlobalRaw)) {
    GlobalRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_pregfx12_atomic_site(site, raw, true, null_saddr);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
