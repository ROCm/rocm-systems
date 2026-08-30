// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1201_target_ops.cpp
/// @brief gfx1201 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_program_analysis_gfx12_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

std::optional<ConSanScratchComponentEncoding>
decode_gfx1201_scratch_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(rdna4::VscratchMachineInst))
    return std::nullopt;
  rdna4::VscratchMachineInst raw{};
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

std::optional<ConSanPrivateComponentEncoding>
decode_gfx1201_private_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(rdna4::VflatMachineInst))
    return std::nullopt;
  rdna4::VflatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0xecu)
    return std::nullopt;
  return ConSanPrivateComponentEncoding{
      .address_vgpr = static_cast<uint16_t>(raw.vaddr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.vsrc),
      .immediate_offset = static_cast<uint32_t>(raw.ioffset),
  };
}

std::optional<ConSanLaneTransferEncoding>
decode_gfx1201_lane_transfer(std::span<const uint8_t> instruction) {
  if (instruction.size() < sizeof(rdna4::Vop3MachineInst))
    return std::nullopt;
  rdna4::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return ConSanLaneTransferEncoding{
      .lane_selector = static_cast<uint32_t>(raw.src1),
      .value_source_operand = 1,
  };
}

bool decode_gfx1201_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  if (mnemonic.starts_with("ds_") && instruction.size() >= sizeof(rdna4::VdsMachineInst)) {
    rdna4::VdsMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_ds_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("flat_atomic") &&
      instruction.size() >= sizeof(rdna4::VflatMachineInst)) {
    rdna4::VflatMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("global_atomic") &&
      instruction.size() >= sizeof(rdna4::VglobalMachineInst)) {
    rdna4::VglobalMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("scratch_atomic") &&
      instruction.size() >= sizeof(rdna4::VscratchMachineInst)) {
    rdna4::VscratchMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_flat_atomic_site(site, raw);
    return true;
  }
  if (mnemonic.starts_with("buffer_atomic") &&
      instruction.size() >= sizeof(rdna4::VbufferMachineInst)) {
    rdna4::VbufferMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    fill_gfx12_buffer_atomic_site(site, raw);
    return true;
  }
  return false;
}

} // namespace rocjitsu::consan_program_analysis_target_detail
