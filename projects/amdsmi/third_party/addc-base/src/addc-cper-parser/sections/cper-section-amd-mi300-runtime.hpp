// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// Parses AMD MI300 runtime error sections.
/// GUID: 32ac0c78-2623-48f6-81a2-ac691780551d
/// Match: GUID only. Per spec (Table 2.1), this GUID is unique to GPU
/// non-standard runtime sections.
/// Note: revision check is omitted because some CPER generators populate
/// program_id with a value other than 0x2 (Instinct).
[[nodiscard]] bool matches_amd_mi300_runtime(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_mi300_runtime(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_mi300_runtime(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
