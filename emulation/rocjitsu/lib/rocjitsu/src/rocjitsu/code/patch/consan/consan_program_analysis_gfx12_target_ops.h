// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx12_target_ops.h
/// @brief Shared gfx12 raw atomic normalization recipes.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

/// Map the gfx12-family cache-scope field into ConSan's causal vocabulary.
///
/// CU and SE are hardware cache domains, not HSA memory scopes. ConSan treats
/// them conservatively as wavefront- and workgroup-local visibility for its
/// detector model; device and system map to agent and system. The exact raw
/// field remains alongside this product. A target's compiler-qualified
/// workgroup acquire is a separate decoded fact because gfx1201 spells it with
/// raw SE while gfx1250 spells it with raw CU.
[[nodiscard]] inline constexpr std::optional<ConSanMemoryScope>
normalize_gfx12_memory_scope(uint32_t raw_scope) {
  switch (raw_scope) {
  case 0:
    return ConSanMemoryScope::Wavefront;
  case 1:
    return ConSanMemoryScope::Workgroup;
  case 2:
    return ConSanMemoryScope::Agent;
  case 3:
    return ConSanMemoryScope::System;
  default:
    return std::nullopt;
  }
}

inline ConSanCacheOperationEncoding
classify_gfx12_cache_operation(std::string_view mnemonic,
                               bool ordinary_acquire_mutation_supported) {
  if (mnemonic == "global_wb")
    return {.operation = ConSanCacheOperation::Release};
  if (mnemonic == "global_inv" || mnemonic == "s_dcache_inv") {
    return {
        .operation = ConSanCacheOperation::Acquire,
        .ordinary_acquire_mutation_supported =
            mnemonic == "global_inv" && ordinary_acquire_mutation_supported,
    };
  }
  return {};
}

inline ConSanWaitInstructionEncoding
classify_gfx12_wait_instruction(std::string_view mnemonic, uint32_t word, rj_code_arch_t arch) {
  const bool bounded_release_counter_form =
      mnemonic == "s_wait_storecnt" || mnemonic == "s_wait_storecnt_dscnt" ||
      mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_loadcnt_dscnt";
  return classify_target_wait_instruction(word, arch, std::nullopt, true,
                                          bounded_release_counter_form);
}

template <typename Raw>
std::optional<ConSanScratchComponentEncoding>
decode_gfx12_scratch_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(Raw))
    return std::nullopt;
  Raw raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0xedu || raw.vaddr != 0u)
    return std::nullopt;
  return ConSanScratchComponentEncoding{
      .vector_address_vgpr = static_cast<uint16_t>(raw.vaddr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.vsrc),
      .scalar_address_sgpr = static_cast<uint16_t>(raw.saddr),
      .immediate_offset = static_cast<uint32_t>(raw.ioffset),
  };
}

template <typename Raw>
std::optional<ConSanLaneTransferEncoding>
decode_gfx12_lane_transfer(std::span<const uint8_t> instruction, int value_source_operand) {
  if (instruction.size() < sizeof(Raw))
    return std::nullopt;
  Raw raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return ConSanLaneTransferEncoding{
      .lane_selector = static_cast<uint32_t>(raw.src1),
      .value_source_operand = value_source_operand,
  };
}

template <typename Raw>
ConSanVectorMemoryEncoding
make_gfx12_vector_memory_encoding(const Raw &raw, uint32_t null_saddr, bool exact_size,
                                  bool ordinary_well_formed, bool ordinary_mutation_supported,
                                  uint32_t workgroup_acquire_scope) {
  return ConSanVectorMemoryEncoding{
      .raw_op = static_cast<uint32_t>(raw.op),
      .raw_saddr = static_cast<uint32_t>(raw.saddr),
      .raw_nv = static_cast<uint32_t>(raw.nv),
      .raw_scale_offset =
          [&] {
            if constexpr (requires { raw.scale_offset; })
              return raw.scale_offset != 0u;
            return false;
          }(),
      .raw_sve = static_cast<uint32_t>(raw.sve),
      .raw_vaddr = static_cast<uint32_t>(raw.vaddr),
      .raw_vsrc = static_cast<uint32_t>(raw.vsrc),
      .raw_vdst = static_cast<uint32_t>(raw.vdst),
      .raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset)),
      .raw_segment = 0u,
      .raw_scope = static_cast<uint32_t>(raw.scope),
      .raw_th = static_cast<uint32_t>(raw.th),
      .scope = normalize_gfx12_memory_scope(static_cast<uint32_t>(raw.scope)),
      .encoded_segment = ConSanEncodedFlatSegment::Unspecified,
      .scalar_provenance_sgpr = raw.saddr == null_saddr
                                    ? std::nullopt
                                    : std::optional<uint16_t>(static_cast<uint16_t>(raw.saddr)),
      .scope_follows_address_space = false,
      .workgroup_acquire_ordering = static_cast<uint32_t>(raw.scope) == workgroup_acquire_scope,
      .exact_size = exact_size,
      .ordinary_well_formed = ordinary_well_formed,
      .ordinary_requires_complete_registers = true,
      .ordinary_mutation_supported = ordinary_mutation_supported,
  };
}

template <typename Raw>
ConSanVectorMemoryDecode decode_gfx12_vector_memory(std::span<const uint8_t> instruction,
                                                    uint32_t null_saddr, uint32_t expected_encoding,
                                                    bool mutation_supported,
                                                    uint32_t workgroup_acquire_scope) {
  if (instruction.size() < sizeof(Raw))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  Raw raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  const bool high_padding = [&] {
    if constexpr (requires { raw.pad_40_48; })
      return raw.pad_40_48 == 0u;
    else
      return raw.pad_40_47 == 0u;
  }();
  const bool exact_size = instruction.size() == sizeof(raw);
  const bool well_formed = exact_size && raw.encoding == expected_encoding && raw.pad_8_13 == 0u &&
                           raw.pad_22_23 == 0u && high_padding && raw.pad_63 == 0u;
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding = make_gfx12_vector_memory_encoding(raw, null_saddr, exact_size, well_formed,
                                                    mutation_supported, workgroup_acquire_scope),
  };
}

template <typename Raw> void fill_gfx12_flat_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  if constexpr (requires { raw.scale_offset; })
    site.raw_scale_offset = raw.scale_offset != 0u;
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_vsrc = static_cast<uint32_t>(raw.vsrc);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.scope = normalize_gfx12_memory_scope(static_cast<uint32_t>(raw.scope));
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.returns_old_value = amdgpu::gfx12_atomic_returns(static_cast<uint8_t>(raw.th));
}

template <typename Raw> void fill_gfx12_buffer_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_vdata = static_cast<uint32_t>(raw.vdata);
  site.raw_rsrc = static_cast<uint32_t>(raw.rsrc);
  site.raw_soffset = static_cast<uint32_t>(raw.soffset);
  site.raw_vaddr = static_cast<uint32_t>(raw.vaddr);
  site.raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
  site.raw_scope = static_cast<uint32_t>(raw.scope);
  site.scope = normalize_gfx12_memory_scope(static_cast<uint32_t>(raw.scope));
  site.raw_th = static_cast<uint32_t>(raw.th);
  site.returns_old_value = amdgpu::gfx12_atomic_returns(static_cast<uint8_t>(raw.th));
}

template <typename Raw> void fill_gfx12_ds_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_addr = static_cast<uint32_t>(raw.addr);
  site.raw_data0 = static_cast<uint32_t>(raw.data0);
  site.raw_data1 = static_cast<uint32_t>(raw.data1);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = static_cast<int32_t>(raw.offset0);
}

template <typename DsRaw, typename FlatRaw, typename GlobalRaw, typename ScratchRaw,
          typename BufferRaw>
bool decode_gfx12_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                              std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("ds_") && instruction.size() >= sizeof(DsRaw)) {
    DsRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_ds_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("flat_atomic") && instruction.size() >= sizeof(FlatRaw)) {
    FlatRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") && instruction.size() >= sizeof(GlobalRaw)) {
    GlobalRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("scratch_atomic") && instruction.size() >= sizeof(ScratchRaw)) {
    ScratchRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("buffer_atomic") && instruction.size() >= sizeof(BufferRaw)) {
    BufferRaw raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_buffer_atomic_site(site, raw);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
