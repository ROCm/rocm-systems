// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx9_cdna_target_ops.cpp
/// @brief Shared gfx942/gfx950 raw pointer-provenance decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_program_analysis_pregfx12_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"

#include <cstring>

namespace rocjitsu::consan_program_analysis_target_detail {

ConSanCacheOperationEncoding classify_gfx9_cdna_cache_operation(std::string_view mnemonic) {
  if (mnemonic == "buffer_wbl2")
    return {.operation = ConSanCacheOperation::Release};
  if (mnemonic == "buffer_inv" || mnemonic == "s_dcache_inv")
    return {.operation = ConSanCacheOperation::Acquire};
  return {};
}

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

ConSanVectorMemoryDecode decode_gfx9_cdna_flat_memory(std::span<const uint8_t> instruction) {
  return decode_pregfx12_vector_memory<cdna4::FlatMachineInst>(instruction, false, 0u);
}

ConSanVectorMemoryDecode decode_gfx9_cdna_global_memory(std::span<const uint8_t> instruction) {
  return decode_pregfx12_vector_memory<cdna4::FlatGlblMachineInst>(instruction, true,
                                                                   kCdnaGlobalNoSaddrEncoding);
}

std::optional<ConSanDirectLdsTransferEncoding>
decode_gfx9_cdna_direct_lds_transfer(std::string_view mnemonic,
                                     std::span<const uint8_t> instruction) {
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
  return ConSanDirectLdsTransferEncoding{.writes_lds = true,
                                         .address_source_operand = std::nullopt};
}

bool decode_gfx9_cdna_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                  std::span<const uint8_t> instruction) {
  return decode_pregfx12_atomic_site<cdna4::FlatMachineInst, cdna4::FlatGlblMachineInst>(
      site, mnemonic, instruction, kCdnaGlobalNoSaddrEncoding);
}

} // namespace rocjitsu::consan_program_analysis_target_detail

namespace rocjitsu {

extern const ConSanProgramAnalysisTargetOperations kConSanGfx9CdnaProgramAnalysisOperations = {
    .classify_cache_operation =
        consan_program_analysis_target_detail::classify_gfx9_cdna_cache_operation,
    .decode_scratch_component =
        consan_program_analysis_target_detail::decode_gfx9_cdna_scratch_component,
    .decode_private_component =
        consan_program_analysis_target_detail::decode_gfx9_cdna_private_component,
    .decode_lane_transfer = consan_program_analysis_target_detail::decode_gfx9_cdna_lane_transfer,
    .decode_accvgpr_transfer =
        consan_program_analysis_target_detail::decode_gfx9_cdna_accvgpr_transfer,
    .decode_flat_memory = consan_program_analysis_target_detail::decode_gfx9_cdna_flat_memory,
    .decode_global_memory = consan_program_analysis_target_detail::decode_gfx9_cdna_global_memory,
    .decode_direct_lds_transfer =
        consan_program_analysis_target_detail::decode_gfx9_cdna_direct_lds_transfer,
    .decode_atomic_site = consan_program_analysis_target_detail::decode_gfx9_cdna_atomic_site,
};

} // namespace rocjitsu
