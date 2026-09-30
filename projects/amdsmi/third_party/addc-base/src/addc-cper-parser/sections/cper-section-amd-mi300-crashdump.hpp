// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// Parses AMD MI300 crashdump sections.
/// GUID: 32ac0c78-2623-48f6-b0d0-7365725fd6ae (shared with EPYC crashdump)
/// Match: same GUID + program_gen == 2 AND ras_gen == 2.
///
/// The ras_gen == 2 restriction makes this mutually exclusive with
/// EPYC classification (program_gen == 2 only when ras_gen == 1).
[[nodiscard]] bool matches_amd_mi300_crashdump(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_mi300_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_mi300_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
