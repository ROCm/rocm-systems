// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx9_cdna_target_ops.cpp
/// @brief Shared gfx942/gfx950 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

std::optional<ConSanScratchComponentEncoding>
decode_gfx9_cdna_scratch_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::FlatScratchMachineInst))
    return std::nullopt;
  cdna4::FlatScratchMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u || raw.addr != 0u)
    return std::nullopt;
  return ConSanScratchComponentEncoding{
      .vector_address_vgpr = static_cast<uint16_t>(raw.addr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.data),
      .scalar_address_sgpr = static_cast<uint16_t>(raw.saddr),
      .immediate_offset = static_cast<uint32_t>(raw.offset),
  };
}

std::optional<ConSanPrivateComponentEncoding>
decode_gfx9_cdna_private_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::FlatMachineInst))
    return std::nullopt;
  cdna4::FlatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u)
    return std::nullopt;
  return ConSanPrivateComponentEncoding{
      .address_vgpr = static_cast<uint16_t>(raw.addr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.data),
      .immediate_offset = static_cast<uint32_t>(raw.offset),
  };
}

std::optional<ConSanLaneTransferEncoding>
decode_gfx9_cdna_lane_transfer(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::Vop3MachineInst))
    return std::nullopt;
  cdna4::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return ConSanLaneTransferEncoding{
      .lane_selector = static_cast<uint32_t>(raw.src1),
      .value_source_operand = 1,
  };
}

std::optional<ConSanAccvgprTransferEncoding>
decode_gfx9_cdna_accvgpr_transfer(std::span<const uint8_t> instruction, bool write_accumulator) {
  if (instruction.size() != sizeof(cdna4::Vop3MachineInst))
    return std::nullopt;
  cdna4::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (write_accumulator)
    return ConSanAccvgprTransferEncoding{.accumulator_vgpr = static_cast<uint16_t>(raw.vdst)};
  if (raw.src0 < 256u || raw.src0 >= 512u)
    return ConSanAccvgprTransferEncoding{.accumulator_vgpr = std::nullopt};
  return ConSanAccvgprTransferEncoding{.accumulator_vgpr = static_cast<uint16_t>(raw.src0 - 256u)};
}

ConSanFlatMemoryDecode decode_gfx9_cdna_flat_memory(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(cdna4::FlatMachineInst))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  cdna4::FlatMachineInst raw{};
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
              .raw_ioffset = static_cast<int32_t>(raw.offset),
              .raw_segment = static_cast<uint32_t>(raw.seg),
              .raw_scope = 0u,
              .raw_th = static_cast<uint32_t>(raw.sc0) | (static_cast<uint32_t>(raw.sc1) << 1u),
              .encoded_segment = segment,
              .scalar_provenance_sgpr = std::nullopt,
              .scope_follows_address_space = true,
              .exact_size = instruction.size() == sizeof(raw),
              .ordinary_well_formed =
                  instruction.size() == sizeof(raw) && raw.encoding == 0x37u && raw.seg == 0u,
              .ordinary_requires_complete_registers = false,
              .ordinary_mutation_supported = false,
          },
  };
}

template <typename Raw>
void fill_gfx9_cdna_flat_atomic_site(ConSanAtomicSite &site, const Raw &raw) {
  site.raw_op = static_cast<uint32_t>(raw.op);
  site.raw_saddr = static_cast<uint32_t>(raw.saddr);
  site.raw_vaddr = static_cast<uint32_t>(raw.addr);
  site.raw_vsrc = static_cast<uint32_t>(raw.data);
  site.raw_vdst = static_cast<uint32_t>(raw.vdst);
  site.raw_scope = 2u;
  site.raw_th = static_cast<uint32_t>(raw.sc0) | (static_cast<uint32_t>(raw.sc1) << 1u);
  site.returns_old_value = raw.sc0 != 0u;
}

bool decode_gfx9_cdna_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                  std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("flat_atomic") && instruction.size() >= sizeof(cdna4::FlatMachineInst)) {
    cdna4::FlatMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx9_cdna_flat_atomic_site(site, raw);
    site.raw_ioffset = static_cast<int32_t>(raw.offset);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") &&
      instruction.size() >= sizeof(cdna4::FlatGlblMachineInst)) {
    cdna4::FlatGlblMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx9_cdna_flat_atomic_site(site, raw);
    site.raw_ioffset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset));
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
