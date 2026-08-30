// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops.h
/// @brief Target-normalized raw operands used by program analysis.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu {

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

} // namespace rocjitsu
