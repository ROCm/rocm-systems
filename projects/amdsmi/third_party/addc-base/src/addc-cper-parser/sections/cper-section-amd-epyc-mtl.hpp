// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

/// Parses AMD EPYC MPx Trace Log (MTL) CPER sections.
/// Matches any GUID with d2==0x1022 AND program_gen=1, ras_gen>=3.
[[nodiscard]] bool matches_amd_epyc_mtl(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_epyc_mtl(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_epyc_mtl(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
