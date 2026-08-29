// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_gfx1250_lds_target_ops.cpp
/// @brief gfx1250 normalized LDS access recipes.

#include "rocjitsu/code/patch/consan/consan_target_lds_ops.h"

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"

#include <climits>

namespace rocjitsu {

std::optional<std::vector<uint32_t>>
consan_build_split_two_address_lds_pair(const ConSanSplitTwoAddressLdsRequest &request,
                                        rj_code_arch_t arch) {
  if (!consan_uses_gfx12_cdna_execution(arch) || request.address_vgpr > 255u ||
      request.first_data_vgpr > 255u || request.second_data_vgpr > 255u ||
      (request.element_dwords == 2u &&
       (request.first_data_vgpr > 254u || request.second_data_vgpr > 254u)) ||
      (request.element_dwords != 1u && request.element_dwords != 2u)) {
    return std::nullopt;
  }
  const uint16_t op =
      request.load ? (request.element_dwords == 1u ? cdna5::kDsLoadB32Vds : cdna5::kDsLoadB64Vds)
                   : (request.element_dwords == 1u ? cdna5::kDsStoreB32Vds : cdna5::kDsStoreB64Vds);
  const uint32_t expected_words =
      4u + 2u * (static_cast<uint32_t>(request.first_byte_offset > UINT16_MAX) +
                 static_cast<uint32_t>(request.second_byte_offset > UINT16_MAX));
  std::vector<uint32_t> words;
  words.reserve(expected_words);
  const auto append_access = [&](uint32_t byte_offset, uint16_t data_vgpr) -> bool {
    uint16_t effective_address_vgpr = request.address_vgpr;
    uint16_t immediate = 0;
    if (byte_offset > UINT16_MAX) {
      if (!request.adjusted_address_vgpr || *request.adjusted_address_vgpr > 255u)
        return false;
      const auto adjust = instrumentation::build_v_add_u32_literal(
          *request.adjusted_address_vgpr, *request.adjusted_address_vgpr, byte_offset,
          request.address_vgpr, arch);
      if (!adjust)
        return false;
      words.insert(words.end(), adjust->begin(), adjust->end());
      effective_address_vgpr = *request.adjusted_address_vgpr;
    } else {
      immediate = static_cast<uint16_t>(byte_offset);
    }
    cdna5::VdsBuilderFields fields{
        .offset0 = static_cast<uint8_t>(immediate),
        .offset1 = static_cast<uint8_t>(immediate >> 8u),
        .addr = static_cast<uint8_t>(effective_address_vgpr),
    };
    if (request.load)
      fields.vdst = static_cast<uint8_t>(data_vgpr);
    else
      fields.data0 = static_cast<uint8_t>(data_vgpr);
    const auto access = cdna5::build_vds(op, fields);
    words.insert(words.end(), access.begin(), access.end());
    return true;
  };
  if (!append_access(request.first_byte_offset, request.first_data_vgpr) ||
      !append_access(request.second_byte_offset, request.second_data_vgpr) ||
      words.size() != expected_words) {
    return std::nullopt;
  }
  return words;
}

} // namespace rocjitsu
