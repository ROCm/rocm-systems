// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1250_target_ops.cpp
/// @brief gfx1250 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_program_analysis_gfx12_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

std::optional<ConSanScratchComponentEncoding>
decode_gfx1250_scratch_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna5::VscratchMachineInst))
    return std::nullopt;
  cdna5::VscratchMachineInst raw{};
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

std::optional<ConSanLaneTransferEncoding>
decode_gfx1250_lane_transfer(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(cdna5::Vop3MachineInst))
    return std::nullopt;
  cdna5::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return ConSanLaneTransferEncoding{
      .lane_selector = static_cast<uint32_t>(raw.src1),
      .value_source_operand = 0,
  };
}

ConSanFlatMemoryDecode decode_gfx1250_flat_memory(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(cdna5::VflatMachineInst))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  cdna5::VflatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  const bool well_formed = instruction.size() == sizeof(raw) && raw.encoding == 0xecu &&
                           raw.pad_8_13 == 0u && raw.pad_22_23 == 0u && raw.pad_40_47 == 0u &&
                           raw.pad_63 == 0u;
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding = make_gfx12_flat_memory_encoding(
          raw, cdna5::OPR_SREG_NULL, instruction.size() == sizeof(raw), well_formed, false),
  };
}

bool decode_gfx1250_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("ds_") && instruction.size() >= sizeof(cdna5::VdsMachineInst)) {
    cdna5::VdsMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_ds_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("flat_atomic") &&
      instruction.size() >= sizeof(cdna5::VflatMachineInst)) {
    cdna5::VflatMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") &&
      instruction.size() >= sizeof(cdna5::VglobalMachineInst)) {
    cdna5::VglobalMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("scratch_atomic") &&
      instruction.size() >= sizeof(cdna5::VscratchMachineInst)) {
    cdna5::VscratchMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("buffer_atomic") &&
      instruction.size() >= sizeof(cdna5::VbufferMachineInst)) {
    cdna5::VbufferMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_buffer_atomic_site(site, raw);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
