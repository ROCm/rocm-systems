// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// Parses AMD MI450 crashdump sections.
/// GUID: 32ac0c78-2623-48f6-b0d0-7365725fd6ae  (shared with EPYC and MI300)
/// Match: same GUID + program_gen == 2 AND ras_gen == 3.
///
/// MI450 uses the GPU-style outer body (valid bits, PCIe_DEVID, PLDM, FW_ID,
/// GPU Error Info) but the register array follows the EPYC crashdump layout
/// (MCA 32x512B + DfOrigWdtAddr + DBG_LOG) rather than MI300X's compact
/// 4-register format.
[[nodiscard]] bool matches_amd_mi450_crashdump(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_mi450_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_mi450_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
