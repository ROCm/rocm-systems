// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_cdna3_cdna4_target_ops.cpp
/// @brief Shared CDNA3/CDNA4 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/targets/shared/consan_program_analysis_cdna3_cdna4_rdna3_common.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan::program_analysis_target_detail {

CacheOperationEncoding classify_cdna3_cdna4_cache_operation(std::string_view mnemonic) {
  if (mnemonic == "buffer_wbl2")
    return {.operation = CacheOperation::Release};
  if (mnemonic == "buffer_inv" || mnemonic == "s_dcache_inv")
    return {.operation = CacheOperation::Acquire};
  return {};
}

WaitInstructionEncoding classify_cdna3_cdna4_wait_instruction(std::string_view, uint32_t word,
                                                              rj_code_arch_t arch) {
  return classify_target_wait_instruction(word, arch, std::nullopt, false, false);
}

std::optional<ScratchComponentEncoding>
decode_cdna3_cdna4_scratch_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::FlatScratchMachineInst))
    return std::nullopt;
  cdna4::FlatScratchMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u || raw.addr != 0u)
    return std::nullopt;
  return ScratchComponentEncoding{
      .vector_address_vgpr = static_cast<uint16_t>(raw.addr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.data),
      .scalar_address_sgpr = static_cast<uint16_t>(raw.saddr),
      .immediate_offset = static_cast<uint32_t>(raw.offset),
  };
}

std::optional<PrivateComponentEncoding>
decode_cdna3_cdna4_private_component(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::FlatMachineInst))
    return std::nullopt;
  cdna4::FlatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u)
    return std::nullopt;
  return PrivateComponentEncoding{
      .address_vgpr = static_cast<uint16_t>(raw.addr),
      .load_data_vgpr = static_cast<uint16_t>(raw.vdst),
      .store_data_vgpr = static_cast<uint16_t>(raw.data),
      .immediate_offset = static_cast<uint32_t>(raw.offset),
  };
}

std::optional<LaneTransferEncoding>
decode_cdna3_cdna4_lane_transfer(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna4::Vop3MachineInst))
    return std::nullopt;
  cdna4::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return LaneTransferEncoding{
      .lane_selector = static_cast<uint32_t>(raw.src1),
      .value_source_operand = 1,
  };
}

std::optional<AccvgprTransferEncoding>
decode_cdna3_cdna4_accvgpr_transfer(std::span<const uint8_t> instruction, bool write_accumulator) {
  if (instruction.size() != sizeof(cdna4::Vop3MachineInst))
    return std::nullopt;
  cdna4::Vop3MachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (write_accumulator)
    return AccvgprTransferEncoding{.accumulator_vgpr = static_cast<uint16_t>(raw.vdst)};
  if (raw.src0 < 256u || raw.src0 >= 512u)
    return AccvgprTransferEncoding{.accumulator_vgpr = std::nullopt};
  return AccvgprTransferEncoding{.accumulator_vgpr = static_cast<uint16_t>(raw.src0 - 256u)};
}

VectorMemoryDecode decode_cdna3_cdna4_flat_memory(std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_vector_memory<cdna4::FlatMachineInst>(instruction, false, 0u);
}

VectorMemoryDecode decode_cdna3_cdna4_global_memory(std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_vector_memory<cdna4::FlatGlblMachineInst>(
      instruction, true, kCdnaGlobalNoSaddrEncoding);
}

std::optional<DirectLdsTransferEncoding>
decode_cdna3_cdna4_direct_lds_transfer(std::string_view mnemonic,
                                       std::span<const uint8_t> instruction) {
  if ((mnemonic == "global_load_lds_dwordx3" || mnemonic == "global_load_lds_dwordx4") &&
      instruction.size() == sizeof(cdna4::FlatGlblMachineInst)) {
    cdna4::FlatGlblMachineInst raw{};
    std::memcpy(&raw, instruction.data(), sizeof(raw));
    if (raw.seg != 2u || raw.vdst != 0u)
      return std::nullopt;
    return DirectLdsTransferEncoding{
        .writes_lds = true,
        .address_source_operand = std::nullopt,
        .memory_address_vgpr = static_cast<uint16_t>(raw.addr),
        .memory_address_vgpr_count = static_cast<uint16_t>(raw.saddr == 0x7fu ? 2u : 1u),
        .m0_address_mask = 0x3fffcu,
        .immediate_byte_offset = sign_extend_13_bit_offset(static_cast<uint32_t>(raw.offset))};
  }
  const bool supported_load =
      mnemonic == "buffer_load_dword" || mnemonic == "buffer_load_dwordx2" ||
      mnemonic == "buffer_load_dwordx3" || mnemonic == "buffer_load_dwordx4";
  if (!supported_load || instruction.size() != sizeof(cdna4::MubufMachineInst))
    return std::nullopt;
  cdna4::MubufMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  // The LDS form reserves VDATA. DWORDX2's physical-lane slot geometry is not
  // yet admitted by the instrumentation contract.
  if (raw.lds == 0u || raw.vdata != 0u || mnemonic == "buffer_load_dwordx2")
    return std::nullopt;
  return DirectLdsTransferEncoding{.writes_lds = true,
                                   .address_source_operand = std::nullopt,
                                   .memory_address_vgpr = static_cast<uint16_t>(raw.vaddr),
                                   .m0_address_mask = 0x3ffffu};
}

bool decode_cdna3_cdna4_atomic_site(AtomicSite &site, std::string_view mnemonic,
                                    std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_atomic_site<cdna4::FlatMachineInst, cdna4::FlatGlblMachineInst>(
      site, mnemonic, instruction, kCdnaGlobalNoSaddrEncoding);
}

} // namespace rocjitsu::consan::program_analysis_target_detail

namespace rocjitsu::consan {

extern const ProgramAnalysisTargetOperations kCdna3Cdna4ProgramAnalysisOperations = {
    .implicit_atomic_width_bits =
        program_analysis_target_detail::implicit_cdna3_cdna4_rdna3_atomic_width_bits,
    .classify_cache_operation =
        program_analysis_target_detail::classify_cdna3_cdna4_cache_operation,
    .classify_wait_instruction =
        program_analysis_target_detail::classify_cdna3_cdna4_wait_instruction,
    .decode_scratch_component =
        program_analysis_target_detail::decode_cdna3_cdna4_scratch_component,
    .decode_private_component =
        program_analysis_target_detail::decode_cdna3_cdna4_private_component,
    .decode_lane_transfer = program_analysis_target_detail::decode_cdna3_cdna4_lane_transfer,
    .decode_accvgpr_transfer = program_analysis_target_detail::decode_cdna3_cdna4_accvgpr_transfer,
    .decode_flat_memory = program_analysis_target_detail::decode_cdna3_cdna4_flat_memory,
    .decode_global_memory = program_analysis_target_detail::decode_cdna3_cdna4_global_memory,
    .decode_direct_lds_transfer =
        program_analysis_target_detail::decode_cdna3_cdna4_direct_lds_transfer,
    .decode_atomic_site = program_analysis_target_detail::decode_cdna3_cdna4_atomic_site,
};

} // namespace rocjitsu::consan
