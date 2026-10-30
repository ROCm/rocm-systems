// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

/// Parses AMD EPYC PMIC fault CPER sections.
/// GUID: a7c9d4f2-6e3b-4b91-9a2f-c8715de4136b, program_gen=1, ras_gen>=3.
[[nodiscard]] bool matches_amd_epyc_pmic(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_epyc_pmic(
    std::span<const uint8_t> body, const SectionDescriptor& desc);

} // namespace addc::cper::sections
