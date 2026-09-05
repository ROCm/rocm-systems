// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

namespace rocjitsu {

namespace {

[[nodiscard]] const ConSanProgramAnalysisTargetOperations *operations(rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target == nullptr ? nullptr : target->program_analysis;
}

template <typename Result, typename... Parameters, typename... Arguments>
[[nodiscard]] Result
invoke_target_operation(rj_code_arch_t arch,
                        Result (*ConSanProgramAnalysisTargetOperations::*operation)(Parameters...),
                        Arguments &&...arguments) {
  const auto *target = operations(arch);
  const auto decoder = target ? target->*operation : nullptr;
  return decoder ? decoder(std::forward<Arguments>(arguments)...) : Result{};
}

} // namespace

uint32_t classify_consan_atomic_width_bits(std::string_view mnemonic, rj_code_arch_t arch) {
  const uint32_t explicit_width = lds_width_bits(mnemonic);
  if (explicit_width != 0u)
    return explicit_width;
  const auto *target = operations(arch);
  return target && target->implicit_atomic_width_bits
             ? target->implicit_atomic_width_bits(mnemonic)
             : 0u;
}

ConSanCacheOperationEncoding classify_consan_cache_operation(std::string_view mnemonic,
                                                             rj_code_arch_t arch) {
  return invoke_target_operation(
      arch, &ConSanProgramAnalysisTargetOperations::classify_cache_operation, mnemonic);
}

ConSanWaitInstructionEncoding classify_consan_wait_instruction(std::string_view mnemonic,
                                                               uint32_t word, rj_code_arch_t arch) {
  return invoke_target_operation(arch,
                                 &ConSanProgramAnalysisTargetOperations::classify_wait_instruction,
                                 mnemonic, word, arch);
}

std::optional<ConSanScratchComponentEncoding>
decode_consan_scratch_component_encoding(std::span<const uint8_t> instruction,
                                         rj_code_arch_t arch) {
  return invoke_target_operation(
      arch, &ConSanProgramAnalysisTargetOperations::decode_scratch_component, instruction);
}

std::optional<ConSanPrivateComponentEncoding>
decode_consan_private_component_encoding(std::span<const uint8_t> instruction,
                                         rj_code_arch_t arch) {
  return invoke_target_operation(
      arch, &ConSanProgramAnalysisTargetOperations::decode_private_component, instruction);
}

std::optional<ConSanLaneTransferEncoding>
decode_consan_lane_transfer_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ConSanProgramAnalysisTargetOperations::decode_lane_transfer,
                                 instruction);
}

std::optional<ConSanAccvgprTransferEncoding>
decode_consan_accvgpr_transfer_index(std::span<const uint8_t> instruction, rj_code_arch_t arch,
                                     bool write_accumulator) {
  return invoke_target_operation(arch,
                                 &ConSanProgramAnalysisTargetOperations::decode_accvgpr_transfer,
                                 instruction, write_accumulator);
}

ConSanVectorMemoryDecode decode_consan_flat_memory_encoding(std::span<const uint8_t> instruction,
                                                            rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ConSanProgramAnalysisTargetOperations::decode_flat_memory,
                                 instruction);
}

ConSanVectorMemoryDecode decode_consan_global_memory_encoding(std::span<const uint8_t> instruction,
                                                              rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ConSanProgramAnalysisTargetOperations::decode_global_memory,
                                 instruction);
}

ConSanBufferMemoryDecode decode_consan_buffer_memory_encoding(std::span<const uint8_t> instruction,
                                                              rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ConSanProgramAnalysisTargetOperations::decode_buffer_memory,
                                 instruction);
}

std::optional<ConSanDirectLdsTransferEncoding> decode_consan_direct_lds_transfer_encoding(
    std::string_view mnemonic, std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch,
                                 &ConSanProgramAnalysisTargetOperations::decode_direct_lds_transfer,
                                 mnemonic, instruction);
}

bool decode_consan_atomic_site_encoding(ConSanAtomicSite &site, std::string_view mnemonic,
                                        std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ConSanProgramAnalysisTargetOperations::decode_atomic_site,
                                 site, mnemonic, instruction);
}

} // namespace rocjitsu
