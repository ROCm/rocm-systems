// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops_internal.h
/// @brief Family-owned raw decoders behind program-analysis target operations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops.h"

namespace rocjitsu::consan_program_analysis_target_detail {

[[nodiscard]] std::optional<ConSanScratchComponentEncoding>
decode_gfx9_cdna_scratch_component(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanPrivateComponentEncoding>
decode_gfx9_cdna_private_component(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanLaneTransferEncoding>
decode_gfx9_cdna_lane_transfer(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanAccvgprTransferEncoding>
decode_gfx9_cdna_accvgpr_transfer(std::span<const uint8_t> instruction, bool write_accumulator);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx9_cdna_flat_memory(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx9_cdna_global_memory(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanDirectLdsTransferEncoding>
decode_gfx9_cdna_direct_lds_transfer(std::string_view mnemonic,
                                     std::span<const uint8_t> instruction);
[[nodiscard]] bool decode_gfx9_cdna_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                                std::span<const uint8_t> instruction);

[[nodiscard]] bool decode_gfx1100_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                              std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1100_flat_memory(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1100_global_memory(std::span<const uint8_t> instruction);

[[nodiscard]] std::optional<ConSanScratchComponentEncoding>
decode_gfx1201_scratch_component(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanPrivateComponentEncoding>
decode_gfx1201_private_component(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanLaneTransferEncoding>
decode_gfx1201_lane_transfer(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1201_flat_memory(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1201_global_memory(std::span<const uint8_t> instruction);
[[nodiscard]] bool decode_gfx1201_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                              std::span<const uint8_t> instruction);

[[nodiscard]] std::optional<ConSanScratchComponentEncoding>
decode_gfx1250_scratch_component(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanLaneTransferEncoding>
decode_gfx1250_lane_transfer(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1250_flat_memory(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanVectorMemoryDecode
decode_gfx1250_global_memory(std::span<const uint8_t> instruction);
[[nodiscard]] ConSanBufferMemoryDecode
decode_gfx1250_buffer_memory(std::span<const uint8_t> instruction);
[[nodiscard]] std::optional<ConSanDirectLdsTransferEncoding>
decode_gfx1250_direct_lds_transfer(std::string_view mnemonic, std::span<const uint8_t> instruction);
[[nodiscard]] bool decode_gfx1250_atomic_site(ConSanAtomicSite &site, std::string_view mnemonic,
                                              std::span<const uint8_t> instruction);

} // namespace rocjitsu::consan_program_analysis_target_detail

namespace rocjitsu {

extern const ConSanProgramAnalysisTargetOperations kConSanGfx9CdnaProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1100ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1201ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1250ProgramAnalysisOperations;

} // namespace rocjitsu
