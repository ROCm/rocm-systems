// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

namespace rocjitsu::consan {

namespace {

[[nodiscard]] const ProgramAnalysisTargetOperations *operations(rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return target == nullptr ? nullptr : target->program_analysis;
}

template <typename Result, typename... Parameters, typename... Arguments>
[[nodiscard]] Result
invoke_target_operation(rj_code_arch_t arch,
                        Result (*ProgramAnalysisTargetOperations::*operation)(Parameters...),
                        Arguments &&...arguments) {
  const auto *target = operations(arch);
  const auto decoder = target ? target->*operation : nullptr;
  return decoder ? decoder(std::forward<Arguments>(arguments)...) : Result{};
}

} // namespace

uint32_t classify_atomic_width_bits(std::string_view mnemonic, rj_code_arch_t arch) {
  const uint32_t explicit_width = lds_width_bits(mnemonic);
  if (explicit_width != 0u)
    return explicit_width;
  const auto *target = operations(arch);
  return target && target->implicit_atomic_width_bits ? target->implicit_atomic_width_bits(mnemonic)
                                                      : 0u;
}

CacheOperationEncoding classify_cache_operation(std::string_view mnemonic, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::classify_cache_operation,
                                 mnemonic);
}

WaitInstructionEncoding classify_wait_instruction(std::string_view mnemonic, uint32_t word,
                                                  rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::classify_wait_instruction,
                                 mnemonic, word, arch);
}

std::optional<ScratchComponentEncoding>
decode_scratch_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_scratch_component,
                                 instruction);
}

std::optional<PrivateComponentEncoding>
decode_private_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_private_component,
                                 instruction);
}

std::optional<LaneTransferEncoding>
decode_lane_transfer_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_lane_transfer,
                                 instruction);
}

std::optional<AccvgprTransferEncoding>
decode_accvgpr_transfer_index(std::span<const uint8_t> instruction, rj_code_arch_t arch,
                              bool write_accumulator) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_accvgpr_transfer,
                                 instruction, write_accumulator);
}

VectorMemoryDecode decode_flat_memory_encoding(std::span<const uint8_t> instruction,
                                               rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_flat_memory,
                                 instruction);
}

VectorMemoryDecode decode_global_memory_encoding(std::span<const uint8_t> instruction,
                                                 rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_global_memory,
                                 instruction);
}

BufferMemoryDecode decode_buffer_memory_encoding(std::span<const uint8_t> instruction,
                                                 rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_buffer_memory,
                                 instruction);
}

std::optional<DirectLdsTransferEncoding>
decode_direct_lds_transfer_encoding(std::string_view mnemonic, std::span<const uint8_t> instruction,
                                    rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_direct_lds_transfer,
                                 mnemonic, instruction);
}

bool decode_atomic_site_encoding(AtomicSite &site, std::string_view mnemonic,
                                 std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  return invoke_target_operation(arch, &ProgramAnalysisTargetOperations::decode_atomic_site, site,
                                 mnemonic, instruction);
}

} // namespace rocjitsu::consan
