// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

/// Parses AMD EPYC Core Debug Dump (CDD) CPER sections.
/// GUID: fdbe4adf-86ec-47e3-89be-693061a0f468, program_gen=1, ras_gen>=3.
[[nodiscard]] bool matches_amd_epyc_cdd(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_epyc_cdd(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_epyc_cdd(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
