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

ConSanCacheOperationEncoding classify_gfx1250_cache_operation(std::string_view mnemonic) {
  return classify_gfx12_cache_operation(mnemonic, false);
}

std::optional<ConSanScratchComponentEncoding>
decode_gfx1250_scratch_component(std::span<const uint8_t> instruction) {
  return decode_gfx12_scratch_component<cdna5::VscratchMachineInst>(instruction);
}

std::optional<ConSanLaneTransferEncoding>
decode_gfx1250_lane_transfer(std::span<const uint8_t> instruction) {
  return decode_gfx12_lane_transfer<cdna5::Vop3MachineInst>(instruction, 0);
}

ConSanVectorMemoryDecode decode_gfx1250_flat_memory(std::span<const uint8_t> instruction) {
  return decode_gfx12_vector_memory<cdna5::VflatMachineInst>(instruction, cdna5::OPR_SREG_NULL,
                                                             0xecu, false, 0u);
}

ConSanVectorMemoryDecode decode_gfx1250_global_memory(std::span<const uint8_t> instruction) {
  return decode_gfx12_vector_memory<cdna5::VglobalMachineInst>(instruction, cdna5::OPR_SREG_NULL,
                                                               0xeeu, false, 0u);
}

ConSanBufferMemoryDecode decode_gfx1250_buffer_memory(std::span<const uint8_t> instruction) {
  if (instruction.size() != sizeof(cdna5::VbufferMachineInst))
    return {.status = ConSanTargetDecodeStatus::UnsupportedEncodingSize, .encoding = {}};
  cdna5::VbufferMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  return {
      .status = ConSanTargetDecodeStatus::Decoded,
      .encoding =
          {
              .raw_ioffset = sign_extend_24(static_cast<uint32_t>(raw.ioffset)),
              .raw_scope = static_cast<uint32_t>(raw.scope),
              .raw_th = static_cast<uint32_t>(raw.th),
              .raw_rsrc = static_cast<uint32_t>(raw.rsrc),
              .raw_soffset = static_cast<uint32_t>(raw.soffset),
              .raw_offen = raw.offen != 0u,
              .raw_idxen = raw.idxen != 0u,
              .raw_vaddr = static_cast<uint32_t>(raw.vaddr),
              .address_sgpr = static_cast<uint16_t>(raw.rsrc),
              .address_vgpr = raw.offen || raw.idxen
                                  ? std::optional<uint16_t>(static_cast<uint16_t>(raw.vaddr))
                                  : std::nullopt,
              .data_vgpr = static_cast<uint16_t>(raw.vdata),
              .well_formed = raw.encoding == 0x31u && raw.pad_8_13 == 0u && raw.pad_23_25 == 0u &&
                             raw.pad_40 == 0u,
          },
  };
}

std::optional<ConSanDirectLdsTransferEncoding>
decode_gfx1250_direct_lds_transfer(std::string_view mnemonic,
                                   std::span<const uint8_t> instruction) {
  const bool async_load = mnemonic.starts_with("global_load_async_to_lds_b");
  const bool async_store = mnemonic.starts_with("global_store_async_from_lds_b");
  if ((!async_load && !async_store) || instruction.size() != sizeof(cdna5::VglobalMachineInst))
    return std::nullopt;
  cdna5::VglobalMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  // Current instrumentation snapshots the explicit LDS VGPR but does not
  // reproduce VGLOBAL's signed immediate, so admit only the exact zero form.
  if (sign_extend_24(static_cast<uint32_t>(raw.ioffset)) != 0)
    return std::nullopt;
  return ConSanDirectLdsTransferEncoding{
      .writes_lds = async_load,
      .address_source_operand = static_cast<uint8_t>(async_load ? 0u : 1u),
  };
}

bool decode_gfx1250_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  return decode_gfx12_atomic_site<cdna5::VdsMachineInst, cdna5::VflatMachineInst,
                                  cdna5::VglobalMachineInst, cdna5::VscratchMachineInst,
                                  cdna5::VbufferMachineInst>(site, mnemonic, instruction);
}

} // namespace rocjitsu::consan_program_analysis_target_detail

namespace rocjitsu {

extern const ConSanProgramAnalysisTargetOperations kConSanGfx1250ProgramAnalysisOperations = {
    .classify_cache_operation =
        consan_program_analysis_target_detail::classify_gfx1250_cache_operation,
    .decode_scratch_component =
        consan_program_analysis_target_detail::decode_gfx1250_scratch_component,
    .decode_lane_transfer = consan_program_analysis_target_detail::decode_gfx1250_lane_transfer,
    .decode_flat_memory = consan_program_analysis_target_detail::decode_gfx1250_flat_memory,
    .decode_global_memory = consan_program_analysis_target_detail::decode_gfx1250_global_memory,
    .decode_buffer_memory = consan_program_analysis_target_detail::decode_gfx1250_buffer_memory,
    .decode_direct_lds_transfer =
        consan_program_analysis_target_detail::decode_gfx1250_direct_lds_transfer,
    .decode_atomic_site = consan_program_analysis_target_detail::decode_gfx1250_atomic_site,
};

} // namespace rocjitsu
