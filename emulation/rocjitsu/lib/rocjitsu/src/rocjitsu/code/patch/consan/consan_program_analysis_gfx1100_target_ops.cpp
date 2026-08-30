// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1100_target_ops.cpp
/// @brief gfx1100 raw program-analysis decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

ConSanVectorMemoryDecode decode_gfx1100_flat_memory(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(rdna3::FlatMachineInst))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  rdna3::FlatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  ConSanEncodedFlatSegment segment = ConSanEncodedFlatSegment::Unspecified;
  if (raw.seg == 1u)
    segment = ConSanEncodedFlatSegment::Private;
  else if (raw.seg == 2u)
    segment = ConSanEncodedFlatSegment::Global;
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding =
          {
              .raw_op = static_cast<uint32_t>(raw.op),
              .raw_saddr = static_cast<uint32_t>(raw.saddr),
              .raw_scale_offset = false,
              .raw_vaddr = static_cast<uint32_t>(raw.addr),
              .raw_vsrc = static_cast<uint32_t>(raw.data),
              .raw_vdst = static_cast<uint32_t>(raw.vdst),
              .raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset)),
              .raw_segment = static_cast<uint32_t>(raw.seg),
              .raw_scope = 0u,
              .raw_th = static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u),
              .encoded_segment = segment,
              .scalar_provenance_sgpr = std::nullopt,
              .scope_follows_address_space = true,
              .exact_size = instruction.size() == sizeof(raw),
              .ordinary_well_formed = instruction.size() == sizeof(raw) && raw.encoding == 0x37u &&
                                      raw.seg == 0u && raw.pad_25 == 0u,
              .ordinary_requires_complete_registers = false,
              .ordinary_mutation_supported = false,
          },
  };
}

ConSanVectorMemoryDecode decode_gfx1100_global_memory(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(rdna3::FlatGlobalMachineInst))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  rdna3::FlatGlobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding =
          {
              .raw_op = static_cast<uint32_t>(raw.op),
              .raw_saddr = static_cast<uint32_t>(raw.saddr),
              .raw_sve = static_cast<uint32_t>(raw.sve),
              .raw_vaddr = static_cast<uint32_t>(raw.addr),
              .raw_vsrc = static_cast<uint32_t>(raw.data),
              .raw_vdst = static_cast<uint32_t>(raw.vdst),
              .raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset)),
              .raw_scope = 2u,
              .raw_th = static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u),
              .scalar_provenance_sgpr =
                  raw.saddr == kRdna3GlobalNoSaddrEncoding
                      ? std::nullopt
                      : std::optional<uint16_t>(static_cast<uint16_t>(raw.saddr)),
              .exact_size = instruction.size() == sizeof(raw),
              .ordinary_well_formed = instruction.size() == sizeof(raw) && raw.encoding == 0x37u &&
                                      raw.seg == 2u && raw.pad_25 == 0u,
              .ordinary_mutation_supported = false,
          },
  };
}

template <typename Raw> void fill_gfx1100_flat_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.addr);
  site.raw_vsrc = static_cast<uint32_t>(raw.data);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset));
  site.raw_scope = 2u;
  site.raw_th = static_cast<uint32_t>(raw.glc) | (static_cast<uint32_t>(raw.slc) << 1u);
  site.returns_old_value = raw.glc != 0u;
}

bool decode_gfx1100_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("flat_atomic") && instruction.size() >= sizeof(rdna3::FlatMachineInst)) {
    rdna3::FlatMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx1100_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") &&
      instruction.size() >= sizeof(rdna3::FlatGlobalMachineInst)) {
    rdna3::FlatGlobalMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx1100_flat_atomic_site(site, raw);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
