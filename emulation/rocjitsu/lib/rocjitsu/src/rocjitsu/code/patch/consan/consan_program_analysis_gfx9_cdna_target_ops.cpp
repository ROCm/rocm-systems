// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx9_cdna_target_ops.cpp
/// @brief Shared gfx942/gfx950 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

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

} // namespace rocjitsu::consan_program_analysis_target_detail
