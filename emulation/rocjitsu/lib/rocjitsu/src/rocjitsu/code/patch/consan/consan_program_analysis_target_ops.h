// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops.h
/// @brief Target-normalized raw operands used by program analysis.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu {

struct ConSanAtomicSite;

struct ConSanScratchComponentEncoding {
  uint16_t vector_address_vgpr = 0;
  uint16_t load_data_vgpr = 0;
  uint16_t store_data_vgpr = 0;
  uint16_t scalar_address_sgpr = 0;
  uint32_t immediate_offset = 0;
};

struct ConSanPrivateComponentEncoding {
  uint16_t address_vgpr = 0;
  uint16_t load_data_vgpr = 0;
  uint16_t store_data_vgpr = 0;
  uint32_t immediate_offset = 0;
};

struct ConSanLaneTransferEncoding {
  uint32_t lane_selector = 0;
  int value_source_operand = 0;
};

struct ConSanAccvgprTransferEncoding {
  std::optional<uint16_t> accumulator_vgpr;
};

enum class ConSanTargetDecodeStatus : uint8_t {
  UnsupportedArchitecture,
  UnsupportedEncodingSize,
  Decoded,
};

enum class ConSanEncodedFlatSegment : uint8_t {
  Unspecified,
  Private,
  Global,
};

/// Target-normalized FLAT fields shared by access and ordinary-memory
/// inventory. The target owner validates raw padding and form bits once;
/// semantic inventory owners decide how the decoded operation is used.
struct ConSanVectorMemoryEncoding {
  uint32_t raw_op = 0;
  uint32_t raw_saddr = 0;
  uint32_t raw_nv = 0;
  bool raw_scale_offset = false;
  uint32_t raw_sve = 0;
  uint32_t raw_vaddr = 0;
  uint32_t raw_vsrc = 0;
  uint32_t raw_vdst = 0;
  int32_t raw_ioffset = 0;
  uint32_t raw_segment = 0;
  uint32_t raw_scope = 0;
  uint32_t raw_th = 0;
  ConSanEncodedFlatSegment encoded_segment = ConSanEncodedFlatSegment::Unspecified;
  std::optional<uint16_t> scalar_provenance_sgpr;
  bool scope_follows_address_space = false;
  bool exact_size = false;
  bool ordinary_well_formed = false;
  bool ordinary_requires_complete_registers = false;
  bool ordinary_mutation_supported = false;
};

struct ConSanVectorMemoryDecode {
  ConSanTargetDecodeStatus status = ConSanTargetDecodeStatus::UnsupportedArchitecture;
  ConSanVectorMemoryEncoding encoding;
};

struct ConSanBufferMemoryEncoding {
  int32_t raw_ioffset = 0;
  uint32_t raw_scope = 0;
  uint32_t raw_th = 0;
  uint32_t raw_rsrc = 0;
  uint32_t raw_soffset = 0;
  bool raw_offen = false;
  bool raw_idxen = false;
  uint32_t raw_vaddr = 0;
  uint16_t address_sgpr = 0;
  std::optional<uint16_t> address_vgpr;
  uint16_t data_vgpr = 0;
  bool well_formed = false;
};

struct ConSanBufferMemoryDecode {
  ConSanTargetDecodeStatus status = ConSanTargetDecodeStatus::UnsupportedArchitecture;
  ConSanBufferMemoryEncoding encoding;
};

struct ConSanDirectLdsTransferEncoding {
  bool writes_lds = false;
  std::optional<uint8_t> address_source_operand;
};

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
