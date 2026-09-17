// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_rdna3_target_ops.cpp
/// @brief rdna3 raw program-analysis decoding.

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

#include "rocjitsu/code/patch/consan/targets/shared/consan_program_analysis_cdna3_cdna4_rdna3_common.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"

namespace rocjitsu::consan::program_analysis_target_detail {

CacheOperationEncoding classify_rdna3_cache_operation(std::string_view mnemonic) {
  if (mnemonic == "s_dcache_inv")
    return {.operation = CacheOperation::Acquire};
  if (mnemonic == "buffer_gl1_inv")
    return {.operation = CacheOperation::AcquirePairPrefix};
  if (mnemonic == "buffer_gl0_inv")
    return {.operation = CacheOperation::AcquirePairCompletion};
  return {};
}

WaitInstructionEncoding classify_rdna3_wait_instruction(std::string_view mnemonic, uint32_t word,
                                                        rj_code_arch_t arch) {
  return classify_target_wait_instruction(word, arch, build_rdna3_s_wait_vscnt0(arch), false,
                                          mnemonic == "s_waitcnt_vscnt");
}

VectorMemoryDecode decode_rdna3_flat_memory(std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_vector_memory<rdna3::FlatMachineInst>(instruction, false, 0u);
}

VectorMemoryDecode decode_rdna3_global_memory(std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_vector_memory<rdna3::FlatGlobalMachineInst>(
      instruction, true, kRdna3GlobalNoSaddrEncoding);
}

bool decode_rdna3_atomic_site(AtomicSite &site, std::string_view mnemonic,
                              std::span<const uint8_t> instruction) {
  return decode_cdna3_cdna4_rdna3_atomic_site<rdna3::FlatMachineInst, rdna3::FlatGlobalMachineInst>(
      site, mnemonic, instruction, kRdna3GlobalNoSaddrEncoding);
}

} // namespace rocjitsu::consan::program_analysis_target_detail

namespace rocjitsu::consan {

extern const ProgramAnalysisTargetOperations kRdna3ProgramAnalysisOperations = {
    .implicit_atomic_width_bits =
        program_analysis_target_detail::implicit_cdna3_cdna4_rdna3_atomic_width_bits,
    .classify_cache_operation = program_analysis_target_detail::classify_rdna3_cache_operation,
    .classify_wait_instruction = program_analysis_target_detail::classify_rdna3_wait_instruction,
    .decode_flat_memory = program_analysis_target_detail::decode_rdna3_flat_memory,
    .decode_global_memory = program_analysis_target_detail::decode_rdna3_global_memory,
    .decode_atomic_site = program_analysis_target_detail::decode_rdna3_atomic_site,
};

} // namespace rocjitsu::consan
