// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops.h
/// @brief Target-normalized raw operands used by program analysis.

#pragma once

#include "rocjitsu/code/patch/consan/consan_program_analysis_encoding.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu::consan {

struct AtomicSite;

/// One target's complete program-analysis normalization facet. Unsupported
/// operations remain null rather than requiring common dispatch code to know
/// which concrete target implements which decoder.
struct ProgramAnalysisTargetOperations {
  /// Width implied by a target-native atomic mnemonic that carries no width
  /// suffix. Returning zero leaves the instruction's width unknown.
  uint32_t (*implicit_atomic_width_bits)(std::string_view) = nullptr;
  CacheOperationEncoding (*classify_cache_operation)(std::string_view) = nullptr;
  WaitInstructionEncoding (*classify_wait_instruction)(std::string_view, uint32_t,
                                                       rj_code_arch_t) = nullptr;
  std::optional<ScratchComponentEncoding> (*decode_scratch_component)(std::span<const uint8_t>) =
      nullptr;
  std::optional<PrivateComponentEncoding> (*decode_private_component)(std::span<const uint8_t>) =
      nullptr;
  std::optional<LaneTransferEncoding> (*decode_lane_transfer)(std::span<const uint8_t>) = nullptr;
  std::optional<AccvgprTransferEncoding> (*decode_accvgpr_transfer)(std::span<const uint8_t>,
                                                                    bool) = nullptr;
  VectorMemoryDecode (*decode_flat_memory)(std::span<const uint8_t>) = nullptr;
  VectorMemoryDecode (*decode_global_memory)(std::span<const uint8_t>) = nullptr;
  BufferMemoryDecode (*decode_buffer_memory)(std::span<const uint8_t>) = nullptr;
  std::optional<DirectLdsTransferEncoding> (*decode_direct_lds_transfer)(
      std::string_view, std::span<const uint8_t>) = nullptr;
  bool (*decode_atomic_site)(AtomicSite &, std::string_view, std::span<const uint8_t>) = nullptr;
};

/// Translate one target-native cache mnemonic into the common synchronization
/// vocabulary. Unsupported targets and non-cache mnemonics return an
/// unsupported operation.
[[nodiscard]] CacheOperationEncoding classify_cache_operation(std::string_view mnemonic,
                                                              rj_code_arch_t arch);

/// Normalize the width encoded by one target-native atomic mnemonic. Generic
/// suffixed spellings are handled once; target providers own any implicit
/// width convention.
[[nodiscard]] uint32_t classify_atomic_width_bits(std::string_view mnemonic, rj_code_arch_t arch);

[[nodiscard]] WaitInstructionEncoding classify_wait_instruction(std::string_view mnemonic,
                                                                uint32_t word, rj_code_arch_t arch);

[[nodiscard]] std::optional<ScratchComponentEncoding>
decode_scratch_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] std::optional<PrivateComponentEncoding>
decode_private_component_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

[[nodiscard]] std::optional<LaneTransferEncoding>
decode_lane_transfer_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch);

/// Decode the raw accumulator operand used by a V_ACCVGPR transfer. Generic
/// instruction operands describe the ordinary VGPR/SGPR side of the transfer;
/// this operation supplies only the architecture-specific accumulator index.
[[nodiscard]] std::optional<AccvgprTransferEncoding>
decode_accvgpr_transfer_index(std::span<const uint8_t> instruction, rj_code_arch_t arch,
                              bool write_accumulator);

[[nodiscard]] VectorMemoryDecode decode_flat_memory_encoding(std::span<const uint8_t> instruction,
                                                             rj_code_arch_t arch);

[[nodiscard]] VectorMemoryDecode decode_global_memory_encoding(std::span<const uint8_t> instruction,
                                                               rj_code_arch_t arch);

[[nodiscard]] BufferMemoryDecode decode_buffer_memory_encoding(std::span<const uint8_t> instruction,
                                                               rj_code_arch_t arch);

[[nodiscard]] std::optional<DirectLdsTransferEncoding>
decode_direct_lds_transfer_encoding(std::string_view mnemonic, std::span<const uint8_t> instruction,
                                    rj_code_arch_t arch);

/// Decode target-native atomic operands into the common atomic inventory
/// product. Returns false when the target or encoded form is not represented;
/// the caller retains its generic decoded operands in that case.
[[nodiscard]] bool decode_atomic_site_encoding(AtomicSite &site, std::string_view mnemonic,
                                               std::span<const uint8_t> instruction,
                                               rj_code_arch_t arch);

} // namespace rocjitsu::consan
