// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops.h
/// @brief Target-normalized raw operands used by program analysis.

#pragma once

#include "rocjitsu/code/patch/consan/consan_program_analysis_encoding.h"
#include "rocjitsu/code/rj_code.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu {

struct ConSanAtomicSite;

/// One target's complete program-analysis normalization facet. Unsupported
/// operations remain null rather than requiring common dispatch code to know
/// which concrete target implements which decoder.
struct ConSanProgramAnalysisTargetOperations {
  /// Width implied by a target-native atomic mnemonic that carries no width
  /// suffix. Returning zero leaves the instruction's width unknown.
  uint32_t (*implicit_atomic_width_bits)(std::string_view) = nullptr;
  ConSanCacheOperationEncoding (*classify_cache_operation)(std::string_view) = nullptr;
  ConSanWaitInstructionEncoding (*classify_wait_instruction)(std::string_view, uint32_t,
                                                             rj_code_arch_t) = nullptr;
  std::optional<ConSanScratchComponentEncoding> (*decode_scratch_component)(
      std::span<const uint8_t>) = nullptr;
  std::optional<ConSanPrivateComponentEncoding> (*decode_private_component)(
      std::span<const uint8_t>) = nullptr;
  std::optional<ConSanLaneTransferEncoding> (*decode_lane_transfer)(std::span<const uint8_t>) =
      nullptr;
  std::optional<ConSanAccvgprTransferEncoding> (*decode_accvgpr_transfer)(std::span<const uint8_t>,
                                                                          bool) = nullptr;
  ConSanVectorMemoryDecode (*decode_flat_memory)(std::span<const uint8_t>) = nullptr;
  ConSanVectorMemoryDecode (*decode_global_memory)(std::span<const uint8_t>) = nullptr;
  ConSanBufferMemoryDecode (*decode_buffer_memory)(std::span<const uint8_t>) = nullptr;
  std::optional<ConSanDirectLdsTransferEncoding> (*decode_direct_lds_transfer)(
      std::string_view, std::span<const uint8_t>) = nullptr;
  bool (*decode_atomic_site)(ConSanAtomicSite &, std::string_view,
                             std::span<const uint8_t>) = nullptr;
};

/// Translate one target-native cache mnemonic into the common synchronization
/// vocabulary. Unsupported targets and non-cache mnemonics return an
/// unsupported operation.
[[nodiscard]] ConSanCacheOperationEncoding
classify_consan_cache_operation(std::string_view mnemonic, rj_code_arch_t arch);

/// Normalize the width encoded by one target-native atomic mnemonic. Generic
/// suffixed spellings are handled once; target providers own any implicit
/// width convention.
[[nodiscard]] uint32_t classify_consan_atomic_width_bits(std::string_view mnemonic,
                                                         rj_code_arch_t arch);

[[nodiscard]] ConSanWaitInstructionEncoding
classify_consan_wait_instruction(std::string_view mnemonic, uint32_t word, rj_code_arch_t arch);

/// One additive target registration. Concrete packages supply an operations
/// facet; the common registry supplies only the architecture key.
template <typename TargetKey> struct ConSanProgramAnalysisTargetRegistrationFor {
  TargetKey target;
  const ConSanProgramAnalysisTargetOperations *operations = nullptr;
};

using ConSanProgramAnalysisTargetRegistration =
    ConSanProgramAnalysisTargetRegistrationFor<rj_code_arch_t>;

template <typename TargetKey>
[[nodiscard]] const ConSanProgramAnalysisTargetOperations *
find_consan_program_analysis_target_operations(
    std::span<const ConSanProgramAnalysisTargetRegistrationFor<TargetKey>> registrations,
    TargetKey target) {
  const auto registration = std::ranges::find(
      registrations, target, &ConSanProgramAnalysisTargetRegistrationFor<TargetKey>::target);
  return registration == registrations.end() ? nullptr : registration->operations;
}

[[nodiscard]] std::optional<ConSanScratchComponentEncoding>
decode_consan_scratch_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] std::optional<ConSanPrivateComponentEncoding>
decode_consan_private_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] std::optional<ConSanLaneTransferEncoding>
decode_consan_lane_transfer_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

/// Decode the raw accumulator operand used by a V_ACCVGPR transfer. Generic
/// instruction operands describe the ordinary VGPR/SGPR side of the transfer;
/// this operation supplies only the architecture-specific accumulator index.
[[nodiscard]] std::optional<ConSanAccvgprTransferEncoding>
decode_consan_accvgpr_transfer_index(std::span<const uint8_t> instruction, rj_code_arch_t arch,
                                     bool write_accumulator);

[[nodiscard]] ConSanVectorMemoryDecode
decode_consan_flat_memory_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] ConSanVectorMemoryDecode
decode_consan_global_memory_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] ConSanBufferMemoryDecode
decode_consan_buffer_memory_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] std::optional<ConSanDirectLdsTransferEncoding>
decode_consan_direct_lds_transfer_encoding(std::string_view mnemonic,
                                           std::span<const uint8_t> instruction,
                                           rj_code_arch_t arch);

/// Decode target-native atomic operands into the common atomic inventory
/// product. Returns false when the target or encoded form is not represented;
/// the caller retains its generic decoded operands in that case.
[[nodiscard]] bool decode_consan_atomic_site_encoding(ConSanAtomicSite &site,
                                                      std::string_view mnemonic,
                                                      std::span<const uint8_t> instruction,
                                                      rj_code_arch_t arch);

} // namespace rocjitsu
