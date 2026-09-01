// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1201_target_ops.cpp
/// @brief gfx1201 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_program_analysis_gfx12_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

std::optional<ConSanScratchComponentEncoding>
decode_gfx1201_scratch_component(std::span<const uint8_t> instruction) {
  return decode_gfx12_scratch_component<rdna4::VscratchMachineInst>(instruction);
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
  // The generated RDNA4 VOP3 operand table keeps the two architectural
  // sources first and appends vdst's lane-preservation use last.
  return decode_gfx12_lane_transfer<rdna4::Vop3MachineInst>(instruction, 0);
}

ConSanVectorMemoryDecode decode_gfx1201_flat_memory(std::span<const uint8_t> instruction) {
  return decode_gfx12_vector_memory<rdna4::VflatMachineInst>(instruction, rdna4::OPR_SREG_NULL,
                                                             0xecu, true, 1u);
}

ConSanVectorMemoryDecode decode_gfx1201_global_memory(std::span<const uint8_t> instruction) {
  return decode_gfx12_vector_memory<rdna4::VglobalMachineInst>(instruction, rdna4::OPR_SREG_NULL,
                                                               0xeeu, true, 1u);
}

bool decode_gfx1201_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  return decode_gfx12_atomic_site<rdna4::VdsMachineInst, rdna4::VflatMachineInst,
                                  rdna4::VglobalMachineInst, rdna4::VscratchMachineInst,
                                  rdna4::VbufferMachineInst>(site, mnemonic, instruction);
}

} // namespace rocjitsu::consan_program_analysis_target_detail

namespace rocjitsu {

extern const ConSanProgramAnalysisTargetOperations kConSanGfx1201ProgramAnalysisOperations = {
    .decode_scratch_component =
        consan_program_analysis_target_detail::decode_gfx1201_scratch_component,
    .decode_private_component =
        consan_program_analysis_target_detail::decode_gfx1201_private_component,
    .decode_lane_transfer = consan_program_analysis_target_detail::decode_gfx1201_lane_transfer,
    .decode_flat_memory = consan_program_analysis_target_detail::decode_gfx1201_flat_memory,
    .decode_global_memory = consan_program_analysis_target_detail::decode_gfx1201_global_memory,
    .decode_atomic_site = consan_program_analysis_target_detail::decode_gfx1201_atomic_site,
};

} // namespace rocjitsu
