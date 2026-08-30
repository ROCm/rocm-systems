// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_gfx1100_target_ops.cpp
/// @brief gfx1100 raw program-analysis decoding.

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/consan_program_analysis_pregfx12_target_ops.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"

namespace rocjitsu::consan_program_analysis_target_detail {

ConSanVectorMemoryDecode decode_gfx1100_flat_memory(std::span<const uint8_t> instruction) {
  return decode_pregfx12_vector_memory<rdna3::FlatMachineInst>(instruction, false, 0u);
}

ConSanVectorMemoryDecode decode_gfx1100_global_memory(std::span<const uint8_t> instruction) {
  return decode_pregfx12_vector_memory<rdna3::FlatGlobalMachineInst>(instruction, true,
                                                                     kRdna3GlobalNoSaddrEncoding);
}

bool decode_gfx1100_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                std::span<const uint8_t> instruction) {
  return decode_pregfx12_atomic_site<rdna3::FlatMachineInst, rdna3::FlatGlobalMachineInst>(
      site, mnemonic, instruction);
}

} // namespace rocjitsu::consan_program_analysis_target_detail
